import torch
import torch.nn as nn


class YoloLSTM(nn.Module):
    def __init__(self, param):
        super(YoloLSTM, self).__init__()

        self.param = param

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(input_size=8192, hidden_size=64, num_layers=2, batch_first=True)
        self.fc_x = nn.Linear(64, 1)
        self.fc_y = nn.Linear(64, 1)

    def forward(self, batch_i):
        batch_x = []
        batch_y = []

        for x in batch_i:
            crop_size, channels, height, width = x.size()

            # クロップ画像の処理
            x = x.view(crop_size, channels, height, width)
            x = self.cnn(x)
            x = x.view(crop_size, -1)

            # LSTM処理をして最後の出力を取得
            _, (h_n, _) = self.lstm(x)
            x = h_n[-1]

            # xとyを回帰
            x_out = self.fc_x(x)
            y_out = self.fc_y(x)
            batch_x.append(x_out)
            batch_y.append(y_out)

        batch_x = torch.stack(batch_x)
        batch_y = torch.stack(batch_y)
        return batch_x, batch_y
