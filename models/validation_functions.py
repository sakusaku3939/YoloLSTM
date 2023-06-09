import torch


# 検証用関数
def get_classification_accuracy(pred, y):
    total = 0
    correct = 0
    _, pred = torch.max(pred.data, 1)
    total += y.size(0)
    correct += (pred == y).sum().item()
    return correct / total
