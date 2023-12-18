import concurrent.futures
import os

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
        img_paths, self.positions = parse_txt(self.pose_txt)
        self.data_dir = os.path.join(root, dataset_name)

        # クロップ画像のパスに変換する
        all_paths = []
        crop_input_queue, crop_output_queue, crop_name_queue = [], [], []
        for path in img_paths:
            dir_name, f_name = path.split("/")
            name = os.path.splitext(f_name)[0]

            input_path = os.path.join(self.data_dir, path)
            output_path = os.path.join(self.data_dir, f"cropped_{dir_name}")
            all_paths.append((input_path, output_path, name))

            # ディレクトリが存在しない場合、YOLOでクロップするキューにも追加する
            if not os.path.isdir(os.path.join(output_path, name)):
                crop_input_queue.append(input_path)
                crop_output_queue.append(output_path)
                crop_name_queue.append(name)

        if len(crop_input_queue) > 0:
            crop_images(input_paths=crop_input_queue, output_paths=crop_output_queue, names=crop_name_queue)

        # ディレクトリ配下にあるクロップ画像のパスを取得
        self.img_paths = []
        for i, data in enumerate(all_paths):
            input_path, output_path, name = data
            file_paths = []
            for current_dir, sub_dirs, files_list in os.walk(os.path.join(output_path, name)):
                for f_name in files_list:
                    file_paths.append(os.path.join(current_dir, f_name))

            self.img_paths.append(file_paths)

    def __getitem__(self, index):
        file_paths = self.img_paths[index]
        images = []

        for f_path in file_paths:
            img = Image.open(os.path.join(self.data_dir, f_path))

            # 画像を前処理
            if self.transform is not None:
                img = self.transform(img)

            images.append(img)

        xy = self.positions[index]
        return images, xy

    def __len__(self):
        return len(self.positions)


def crop_images(input_paths, output_paths, names):
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        list(tqdm(
            executor.map(yolo, input_paths, output_paths, names),
            total=len(names),
        ))


def yolo(input_path, output_path, name):
    model = YOLO("yolov8x.pt")
    model(input_path, project=output_path, name=name, save_crop=True, conf=0.1)
    # 検出物体が存在せず、フォルダが作成されなかった場合は、空のフォルダを作成する
    os.makedirs(os.path.join(output_path, name), exist_ok=True)


def collate_fn(batch_list):
    images_list = [data[0] for data in batch_list]
    label_list = [data[1] for data in batch_list]

    # batchリストを1つのTensorにまとめる
    images = [torch.stack(cropped if len(cropped) > 0 else [torch.ones(3, 64, 64)]) for cropped in images_list]
    labels = torch.tensor(np.array(label_list))

    return images, labels
