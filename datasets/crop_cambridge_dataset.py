import concurrent.futures
import glob
import os
import re
import itertools

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from ultralytics import YOLO

from datasets.cambridge_dataset import parse_txt


class CropCambridgeDataset(Dataset):
    def __init__(self, root, dataset_name, txt_path, transform):
        self.transform = transform

        self.pose_txt = os.path.join(root, dataset_name, txt_path)
        self.img_paths, self.positions = parse_txt(self.pose_txt)
        self.data_dir = os.path.join(root, dataset_name)

        input_paths, output_paths, names = [], [], []
        for path in self.img_paths:
            dir_name, f_name = path.split("/")
            # ディレクトリが存在しない場合、YOLOでクロップするキューに追加する
            if not os.path.isdir(os.path.join(self.data_dir, f"cropped_{dir_name}", os.path.splitext(f_name)[0])):
                input_paths.append(os.path.join(self.data_dir, path))
                output_paths.append(os.path.join(self.data_dir, f"cropped_{dir_name}"))
                names.append(os.path.splitext(f_name)[0])

        crop_images(input_paths=input_paths, output_paths=output_paths, names=names)

        # cropped_paths = glob.glob(f"{root}/cropped_*")
        # self.dataset = []
        #
        # # datasetにラベルと画像パスを追加
        # for i, c_path in enumerate(cropped_paths):
        #     dir_names = sorted(os.listdir(c_path))
        #     for d_name in dir_names:
        #         file_paths = []
        #
        #         # ディレクトリ配下にあるクロップ画像のパスを取得
        #         for current_dir, sub_dirs, files_list in os.walk(f"{c_path}/{d_name}"):
        #             for f_name in files_list:
        #                 file_paths.append(os.path.join(current_dir, f_name))
        #
        #         position = [float(p) for p in re.findall(r'\d+', c_path)]
        #         self.dataset.append({"xy": position, "paths": file_paths})

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(os.path.join(self.data_dir, img_path))

        if self.transform is not None:
            img = self.transform(img)

        xy = self.positions[index]
        return img, torch.tensor(xy)

    def __len__(self):
        return len(self.img_paths)


def crop_images(input_paths, output_paths, names):
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        list(tqdm(
            executor.map(yolo, input_paths, output_paths, names),
            total=len(names)
        ))
        executor.shutdown(wait=True)


def yolo(input_path, output_path, name):
    model = YOLO("yolov8x.pt")
    model(input_path, project=output_path, name=name, save_crop=True, conf=0.1)
    # 検出物体が存在せず、フォルダが作成されなかった場合は、空のフォルダを作成する
    os.makedirs(os.path.join(output_path, name), exist_ok=True)


def collate_fn(batch_list):
    images_list = [data[0] for data in batch_list]
    label_list = [data[1] for data in batch_list]

    # batchリストを1つのTensorにまとめる
    images = [torch.stack(batch) for batch in images_list]
    labels = torch.tensor(label_list)

    return images, labels
