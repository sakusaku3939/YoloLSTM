import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class CambridgeDataset(Dataset):
    def __init__(self, root, dataset_name, txt_path, transform):
        self.transform = transform

        self.pose_txt = os.path.join(root, dataset_name, txt_path)
        self.img_paths, self.positions = self.parse_txt(self.pose_txt)
        self.data_dir = os.path.join(root, dataset_name)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(os.path.join(self.data_dir, img_path))

        if self.transform is not None:
            img = self.transform(img)

        xy = self.positions[index]
        return img, torch.tensor(xy)

    def __len__(self):
        return len(self.img_paths)

    def parse_txt(self, fpath):
        positions = []
        image_paths = []
        f = open(fpath)

        for line in f.readlines()[3::]:  # ヘッダー部分の3行をスキップする
            cur = line.split(' ')
            xy = np.array([float(v) for v in cur[1:3]], dtype=np.float32)
            image_paths.append(cur[0])
            positions.append(xy)
        f.close()
        return image_paths, positions
