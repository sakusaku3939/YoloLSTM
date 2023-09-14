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
        self.lstm = nn.LSTM(input_size=8192, hidden_size=2048, num_layers=2, batch_first=True)
        self.fc = nn.Linear(2048, 3)

    def forward(self, batch_i):
        batch_out = []

        for x in batch_i:
            crop_size, channels, height, width = x.size()

            # クロップ画像の処理
            x = x.view(crop_size, channels, height, width)
            x = self.cnn(x)
            x = x.view(crop_size, -1)

            # LSTM処理をして最後の出力を取得
            _, (h_n, _) = self.lstm(x)
            x = h_n[-1]

            # 2D座標を回帰
            x = self.fc(x)
            batch_out.append(x)

        batch_out = torch.stack(batch_out)
        return batch_out
