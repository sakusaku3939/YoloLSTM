import torch


# 検証用関数 pred: 推測値, label: 正解データ
def get_classification_accuracy(pred, label):
    total = 0
    correct = 0
    # # 各行から最大値を選んで、最大値のindexを格納する
    # _, pred = torch.max(pred.data, dim=1)
    # # Tensorの0次元目のサイズを取得
    # total += label.size(0)
    # # sum()で True の合計値を算出し、item()で numpy.int64 から int型 の数値に変換
    # correct += (pred == labels).sum().item()
    return pred == label
