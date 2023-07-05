import glob
import os
from torch.utils.data import Dataset
from PIL import Image
from ultralytics import YOLO


class CropDataset(Dataset):
    def __init__(self, root, transform) -> None:
        super().__init__()

        # 初回実行の場合は、画像をYOLOでクロップする
        if not glob.glob(root + "/cropped_*"):
            for f in glob.glob(root + "/*"):
                crop_images(input_path=f, output_path=root, name="cropped_" + os.path.split(f)[1])

        cropped_paths = glob.glob(root + "/cropped_*")
        self.dataset = []

        # ディレクトリ配下にあるクロップ画像のパスを取得
        for i, c_path in enumerate(cropped_paths):
            for current_dir, sub_dirs, files_list in os.walk(c_path):
                for file_name in files_list:
                    self.dataset.append({"label": i, "file_path": os.path.join(current_dir, file_name)})

        self.root = root
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.dataset[index]["file_path"]
        img = Image.open(file_path)

        # 画像を前処理
        if self.transform is not None:
            img = self.transform(img)

        label = self.dataset[index]["label"]
        return img, label, file_path

    def __len__(self):
        return len(self.dataset)


def crop_images(input_path, output_path, name):
    model = YOLO("yolov8n.pt")
    model(input_path, project=output_path, name=name, save_crop=True, conf=0.1)
