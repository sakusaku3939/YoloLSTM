import torch


# 検証用関数 pred: 推測値, labels: 正解データ
def get_classification_accuracy(pred, labels):
    total = 0
    correct = 0
    # 各行の最大値を抽出して1次元に減らす
    _, pred = torch.max(pred.data, 1)
    # Tensorの0次元目のサイズを取得
    total += labels.size(0)
    # sum()で True の合計値を算出し、item()で numpy.int64 から int型 の数値に変換
    correct += (pred == labels).sum().item()
    return correct / total
