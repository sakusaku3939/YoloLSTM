import torch
import torchvision
import torchvision.transforms as transforms
from config import get_config
import sys

from helper.image_dataset import ImageDataset

sys.path.append('../')


def load_image():
    batch_size = get_config("general", "batch_size")
    num_workers = get_config("general", "num_workers")

    # データの前処理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # データセットの読み込み
    dataset = ImageDataset(transform)

    # 割合から個数を出す
    train_ratio = 60
    train_set = int(len(dataset) * train_ratio / 100)
    test_set = int(len(dataset) - train_set)

    # 学習データと検証データに分割
    train_set, test_set = torch.utils.data.random_split(dataset, [train_set, test_set])

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


def load_test_image():
    batch_size = get_config("general", "batch_size")
    num_workers = get_config("general", "num_workers")

    # データの前処理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # データセットの読み込み
    dataset = torchvision.datasets.ImageFolder("./data/test", transform)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return test_loader
