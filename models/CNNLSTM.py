import torch
import torch.nn as nn


# モデルの定義
class CNNLSTM(nn.Module):
    def __init__(self, param):
        super(CNNLSTM, self).__init__()

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
        self.fc = nn.Linear(64, 2)

    def forward(self, batch_i):
        batch_o = []
        for x in batch_i:
            crop_size, channels, height, width = x.size()

            # クロップ画像の処理
            x = x.view(crop_size, channels, height, width)
            x = self.cnn(x)
            x = x.view(crop_size, -1)

            # LSTM処理をして最後の出力を取得
            _, (h_n, _) = self.lstm(x)
            x = h_n[-1]

            # LSTM層の出力から2値に分類
            x = self.fc(x)
            batch_o.append(x)

        batch_o = torch.stack(batch_o)
        return batch_o
