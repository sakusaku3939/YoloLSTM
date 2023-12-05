import torch
from torch import nn
import torch.nn.functional as F


class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.lrn = nn.LocalResponseNorm(5)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                                   nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU())

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.lrn(x)
        x = self.conv2(x)
        x = self.lrn(x)
        x = self.max_pool(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.max_pool(x)
        x = self.inception4a(x)
        aux1 = x

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = x

        x = self.inception4e(x)
        x = self.max_pool(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        aux3 = x

        if self.training:
            return aux1, aux2, aux3

        return aux3


class Inception(nn.Module):
    def __init__(
            self,
            in_channels,
            ch1x1,
            ch3x3red,
            ch3x3,
            ch5x5red,
            ch5x5,
            pool_proj,
    ):
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat([branch1, branch2, branch3, branch4], 1)

        return out


class InceptionAux(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            dropout,
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (4, 4))  # (N, 512 or 528, 4, 4)
        x = self.conv(x)  # (N, 128, 4, 4)
        x = torch.flatten(x, 1)  # (N, 128 * 4 * 4)
        x = self.fc1(x)  # (N, 1024)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (N, num_classes)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


def calc_loss(outputs, labels):
    loss_function = nn.MSELoss()

    loss1 = loss_function(outputs[0], labels)
    loss2 = loss_function(outputs[1], labels)
    loss3 = loss_function(outputs[2], labels)

    loss = 0.3 * loss1 + 0.3 * loss2 + loss3
    return loss
