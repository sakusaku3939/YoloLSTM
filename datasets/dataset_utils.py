import torch
import torchvision.transforms as transforms
import sys

from datasets.cambridge_dataset import CambridgeDataset
from datasets.crop_cambridge_dataset import CropCambridgeDataset, collate_fn
from datasets.image_dataset import ImageDataset

sys.path.append('../')


def load_cropped_image(batch_size, num_workers, random_state):
    # データの前処理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # データセットの読み込み
    train_set = CropCambridgeDataset(
        "D:/LocalizationDatasets/CambridgeLandmarks",
        "ShopFacade",
        "dataset_train.txt",
        transform,
    )
    valid_set = CropCambridgeDataset(
        "D:/LocalizationDatasets/CambridgeLandmarks",
        "ShopFacade",
        "dataset_test.txt",
        transform,
    )

    # 乱数シードの固定
    g = torch.Generator()
    g.manual_seed(random_state)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        collate_fn=collate_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader


def load_image(batch_size, num_workers, random_state):
    # データの前処理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # データセットの読み込み
    train_set = CambridgeDataset(
        "D:/LocalizationDatasets/CambridgeLandmarks",
        "ShopFacade",
        "dataset_train.txt",
        transform,
    )
    valid_set = CambridgeDataset(
        "D:/LocalizationDatasets/CambridgeLandmarks",
        "ShopFacade",
        "dataset_test.txt",
        transform,
    )

    # 乱数シードの固定
    g = torch.Generator()
    g.manual_seed(random_state)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        generator=g,
    )

    return train_loader, valid_loader


def load_cropped_test_image(batch_size, num_workers):
    # データの前処理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # データセットの読み込み
    dataset = CropDataset("data_all/test", transform, num_workers)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return test_loader


def load_test_image(batch_size, num_workers):
    # データの前処理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # データセットの読み込み
    dataset = ImageDataset("./cnn_data_all/test", transform)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return test_loader
