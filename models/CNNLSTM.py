import torch.nn as nn


# モデルの定義
class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
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

    def forward(self, x):
        batch_size, num_images, channels, height, width = x.size()
        # 画像データの処理
        x = x.view(batch_size * num_images, channels, height, width)
        x = self.cnn(x)
        x = x.view(batch_size, num_images, -1)
        # LSTM処理
        _, (h_n, _) = self.lstm(x)
        # 最後のLSTM層の出力を取得
        x = h_n[-1]
        x = self.fc(x)
        return x
