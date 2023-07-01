import torch
from torch.utils.data import Dataset
from PIL import Image
import csv

path = "./data/train/0627/"


class ImageDataset(Dataset):
    def __init__(self, transform) -> None:
        super().__init__()
        # CSVファイルから画像パスと正解データに分割
        with open(path + "label.csv") as f:
            csv_data = csv.reader(f)
            data = [line for line in csv_data]
            image_paths = [line[0] for line in data]
            labels = [torch.tensor([float(line[1]), float(line[2])], dtype=torch.float32) for line in data]

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = Image.open(path + image_path)

        # 画像を前処理
        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[index]
        return img, label, image_path

    def __len__(self):
        return len(self.image_paths)
