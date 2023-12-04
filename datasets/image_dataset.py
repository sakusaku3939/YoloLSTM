import glob
import os
import re

import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root, transform) -> None:
        super().__init__()
        self.dataset = []

        # datasetにラベルと画像パスを追加
        dir_names = sorted(os.listdir(root))
        for d_name in dir_names:
            position = [float(p) for p in re.findall(r'\d+', d_name)]
            file_paths = glob.glob(f"{root}/{d_name}/*")
            for f_path in file_paths:
                self.dataset.append({"target": position, "f_path": f_path})

        self.root = root
        self.transform = transform

    def __getitem__(self, index):
        f_path = self.dataset[index]["f_path"]

        img = Image.open(f_path)

        # 画像を前処理
        if self.transform is not None:
            img = self.transform(img)

        target = self.dataset[index]["target"]
        return img, torch.tensor(target)

    def __len__(self):
        return len(self.dataset)

