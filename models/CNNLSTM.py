import torch
import torch.nn as nn

from models.GoogLeNet import GoogLeNet

"""
MIT License
Copyright (c) 2019 Qunjie Zhou

https://github.com/GrumpyZhou/visloc-apr

@InProceedings{Walch2017ICCV,
  title = {Image-Based Localization Using LSTMs for Structured Feature Correlation},
  author = {Walch, Florian and Hazirbas, Caner and Leal-Taixe, Laura and Sattler, Torsten and Hilsenbeck, Sebastian and Cremers, Daniel},
  booktitle = {ICCV},
  year = {2017}
}
"""


class CNNLSTM(nn.Module):
    def __init__(self, param):
        super(CNNLSTM, self).__init__()
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
        self.regress_fc_pose = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU())
        self.lstm4dir = FourDirectionalLSTM(seq_size=32, origin_feat_size=2048, hidden_size=256)
        self.regress_lstm4d = nn.Sequential(self.lstm4dir, nn.Dropout(0.5))
        self.regress_fc_xyz = nn.Linear(1024, 6)

    def forward(self, x):
        x = self.regress_fc_pose(x)
        x = self.regress_lstm4d(x)
        xyz = self.regress_fc_xyz(x)
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
        """Return initialized hidden states and cell states for each biodirectional lstm cell"""
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
