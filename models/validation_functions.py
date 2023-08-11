import torch
from sklearn.metrics import r2_score


# 検証用関数 pred: 推測値, target: 正解データ
def get_r2_accuracy(pred, target):
    # print(pred, target)
    return r2_score(target.detach().cpu(), pred.detach().cpu())


def get_classification_accuracy(pred, labels):
    total = 0
    correct = 0

    # 各行から最大値を選んで、最大値のindexを格納する
    _, pred = torch.max(pred.data, dim=1)
    # Tensorの0次元目のサイズを取得
    total += labels.size(0)
    # sum()で indexが等しい要素 の合計値を算出し、item()で numpy.int64 から int型 の数値に変換
    correct += (pred == labels).sum().item()

    return correct / total
