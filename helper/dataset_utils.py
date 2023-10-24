import torch
import torchvision.transforms as transforms
from config import get_config
import sys

from helper.crop_dataset import CropDataset, collate_fn
from helper.image_dataset import ImageDataset

sys.path.append('../')


def load_cropped_image():
    batch_size = get_config("general", "batch_size")
    num_workers = get_config("general", "num_workers")

    # データの前処理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # データセットの読み込み
    train_set = CropDataset("data/train", transform)
    valid_set = CropDataset("data/valid", transform)

    # 乱数シードの固定
    random_state = get_config("general", "random_state")
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


def load_image():
    batch_size = get_config("general", "batch_size")
    num_workers = get_config("general", "num_workers")

    # データの前処理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # データセットの読み込み
    train_set = ImageDataset("./cnn_data/train", transform)
    valid_set = ImageDataset("./cnn_data/valid", transform)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, valid_loader


def load_cropped_test_image():
    batch_size = get_config("general", "batch_size")
    num_workers = get_config("general", "num_workers")

    # データの前処理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # データセットの読み込み
    dataset = CropDataset("data/test", transform)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return test_loader


def load_test_image():
    batch_size = get_config("general", "batch_size")
    num_workers = get_config("general", "num_workers")

    # データの前処理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # データセットの読み込み
    dataset = ImageDataset("./cnn_data/test", transform)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return test_loader