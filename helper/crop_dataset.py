import glob
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


class CropDataset(Dataset):
    def __init__(self, root, transform) -> None:
        super().__init__()

        # 初回実行の場合は、画像をYOLOでクロップする
        if not glob.glob(f"{root}/cropped_*"):
            for f in glob.glob(f"{root}/*"):
                crop_images(input_path=f, output_path=f"{root}/cropped_{os.path.split(f)[1]}")

        cropped_paths = glob.glob(f"{root}/cropped_*")
        self.dataset = []

        # ディレクトリ配下にあるクロップ画像のパスを取得
        for i, c_path in enumerate(cropped_paths):
            dir_names = sorted(os.listdir(c_path))
            for d_name in dir_names:
                file_paths = []
                for current_dir, sub_dirs, files_list in os.walk(f"{c_path}/{d_name}"):
                    for f_name in files_list:
                        file_paths.append(os.path.join(current_dir, f_name))
                self.dataset.append({"label": i, "file_paths": file_paths})

        self.root = root
        self.transform = transform

    def __getitem__(self, index):
        file_paths = self.dataset[index]["file_paths"]
        images = []

        for f_path in file_paths:
            img = Image.open(f_path)

            # 画像を前処理
            if self.transform is not None:
                img = self.transform(img)

            images.append(img)

        label = self.dataset[index]["label"]
        return images, label

    def __len__(self):
        return len(self.dataset)


def crop_images(input_path, output_path):
    model = YOLO("yolov8n.pt")
    file_names = sorted(os.listdir(input_path))

    for f_name in tqdm(file_names):
        name = os.path.splitext(f_name)[0]
        model(f"{input_path}/{f_name}", project=output_path, name=name, save_crop=True, conf=0.1)


def collate_fn(batch_list):
    images_list = [data[0] for data in batch_list]
    label_list = [data[1] for data in batch_list]

    # batchリストを1つのTensorにまとめる
    images = [torch.stack(batch) for batch in images_list]
    labels = torch.tensor(label_list)

    return images, labels