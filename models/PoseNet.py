import torch.nn as nn

from models.base.GoogLeNet import GoogLeNet

"""
MIT License
Copyright (c) 2019 Qunjie Zhou

https://github.com/GrumpyZhou/visloc-apr

@Inproceedings{Kendall2015ICCV,
  Title                    = {PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization},
  Author                   = {Kendall, Alex and Grimes, Matthew and Cipolla, Roberto},
  Booktitle                = {ICCV},
  Year                     = {2015},

  Optorganization          = {IEEE},
  Optpages                 = {2938--2946}
}
"""


class PoseNet(nn.Module):
    def __init__(self, param):
        super(PoseNet, self).__init__()
        self.param = param

        self.extract = GoogLeNet()
        self.regress1 = Regression('regress1')
        self.regress2 = Regression('regress2')
        self.regress3 = Regression('regress3')

    def forward(self, x):
        if self.training:
            feat4a, feat4d, feat5b = self.extract(x)
            pose = [self.regress1(feat4a), self.regress2(feat4d), self.regress3(feat5b)]
        else:
            feat5b = self.extract(x)
            pose = self.regress3(feat5b)
        return pose


class Regression(nn.Module):
    def __init__(self, regid):
        super(Regression, self).__init__()
        conv_in = {"regress1": 512, "regress2": 528}

        if regid != "regress3":
            self.projection = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=3),
                                            nn.Conv2d(conv_in[regid], 128, kernel_size=1),
                                            nn.ReLU())
            self.regress_fc_pose = nn.Sequential(nn.Linear(2048, 1024),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.7))
            self.regress_fc_xy = nn.Linear(1024, 2)
        else:
            self.projection = nn.AvgPool2d(kernel_size=7, stride=1)
            self.regress_fc_pose = nn.Sequential(nn.Linear(1024, 2048),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.5))
            self.regress_fc_xy = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.projection(x)
        x = self.regress_fc_pose(x.view(x.size(0), -1))
        xy = self.regress_fc_xy(x)
        return xy
