import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GoogLeNet import GoogLeNet


class CNNLSTM(nn.Module):
    def __init__(self, param):
        super(CNNLSTM, self).__init__()
        self.param = param

        self.extract = GoogLeNet()
        self.regress1 = Regression('regress1')
        self.regress2 = Regression('regress2')
        self.regress3 = Regression('regress3')

        # # Loss params
        # self.learn_weighting = config.learn_weighting
        # if self.learn_weighting:
        #     # Learned loss weighting during training
        #     sx, sq = config.homo_init
        #     # Variances variables to learn
        #     self.sx = nn.Parameter(torch.tensor(sx))
        #     self.sq = nn.Parameter(torch.tensor(sq))
        # else:
        #     # Fixed loss weighting with beta
        #     self.beta = config.beta
        #
        # self.to(self.device)
        # self.init_weights_(config.weights_dict)
        # self.set_optimizer_(config)

    def forward(self, x):
        if self.training:
            feat4a, feat4d, feat5b = self.extract(x)
            # pose = [self.regress1(feat4a), self.regress2(feat4d), self.regress3(feat5b)]
            pose = [feat4a, feat4d, feat5b]
            # print(pose)
        else:
            pose = self.extract(x)
            # pose = self.regress3(feat5b)
            # print(pose)
            # feat5b = self.extract(x)
            # print(feat5b)
            # pose = self.regress3(feat5b)
            # print(pose)
        # feat4a, feat4d, feat5b = self.extract(x)
        # pose = self.regress3(feat5b)
        return pose


class Regression(nn.Module):
    def __init__(self, regid):
        super(Regression, self).__init__()
        conv_in = {"regress1": 512, "regress2": 528}
        if regid != "regress3":
            # self.projection = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=3),
            #                                 nn.Conv2d(conv_in[regid], 128, kernel_size=1),
            #                                 nn.ReLU())
            # self.projection = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
            self.regress_fc_pose = nn.Sequential(nn.Linear(conv_in[regid], 1024), nn.ReLU())
            self.lstm4dir = FourDirectionalLSTM(seq_size=32, origin_feat_size=1024, hidden_size=256)
            self.regress_lstm4d = nn.Sequential(self.lstm4dir, nn.Dropout(0.7))
            self.regress_fc_xyz = nn.Linear(1024, 6)
            # self.regress_fc_wpqr = nn.Linear(1024, 4)
        else:
            # self.projection = nn.AvgPool2d(kernel_size=7, stride=1)
            self.regress_fc_pose = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU())
            self.lstm4dir = FourDirectionalLSTM(seq_size=32, origin_feat_size=2048, hidden_size=256)
            self.regress_lstm4d = nn.Sequential(self.lstm4dir, nn.Dropout(0.5))
            self.regress_fc_xyz = nn.Linear(1024, 6)
            # self.regress_fc_wpqr = nn.Linear(1024, 4)

    def forward(self, x):
        # x = self.projection(x)
        x = self.regress_fc_pose(x.view(x.size(0), -1))
        x = self.regress_lstm4d(x)
        xyz = self.regress_fc_xyz(x)
        # wpqr = self.regress_fc_wpqr(x)
        # wpqr = F.normalize(wpqr, p=2, dim=1)
        return xyz


class FourDirectionalLSTM(nn.Module):
    def __init__(self, seq_size, origin_feat_size, hidden_size):
        super(FourDirectionalLSTM, self).__init__()
        self.feat_size = origin_feat_size // seq_size
        self.seq_size = seq_size
        self.hidden_size = hidden_size
        self.lstm_rightleft = nn.LSTM(self.feat_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm_downup = nn.LSTM(self.seq_size, self.hidden_size, batch_first=True, bidirectional=True)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return (torch.randn(2, batch_size, self.hidden_size).to(device),
                torch.randn(2, batch_size, self.hidden_size).to(device))

    def forward(self, x):
        batch_size = x.size()[0]
        x_rightleft = x.view(batch_size, self.seq_size, self.feat_size)
        x_downup = x_rightleft.transpose(1, 2)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)
        out_rightleft, hidden_rightleft = self.lstm_rightleft(x_rightleft, hidden_rightleft)
        out_downup, hidden_downup = self.lstm_downup(x_downup, hidden_downup)
        hlr_fw = hidden_rightleft[0][0].squeeze()
        hlr_bw = hidden_rightleft[0][1].squeeze()
        hud_fw = hidden_downup[0][0].squeeze()
        hud_bw = hidden_downup[0][1].squeeze()
        return torch.cat([hlr_fw, hlr_bw, hud_fw, hud_bw], dim=1)
