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
        self.fc = nn.Linear(64, 1)

    def forward(self, inputs):
        outputs = []
        for x in inputs:
            batch_size, channels, height, width = x.size()

            # 画像データの処理
            x = x.view(batch_size, channels, height, width)
            x = self.cnn(x)
            x = x.view(batch_size, -1)

            # LSTM処理をして最後の出力を取得
            _, (h_n, _) = self.lstm(x)
            x = h_n[-1]

            # LSTM層の出力から位置を推定
            x = self.fc(x)
            outputs.append(x)

        print(outputs[0].dtype)
        outputs = torch.cat(outputs)
        print(outputs)
        return outputs
