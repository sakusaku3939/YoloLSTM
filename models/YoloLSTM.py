import torch
import torch.nn as nn


class YoloLSTM(nn.Module):
    def __init__(self, param):
        super(YoloLSTM, self).__init__()

        self.param = param

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 12, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(input_size=2028, hidden_size=2028, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2028, 2)

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
