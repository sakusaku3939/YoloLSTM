import os

import cv2
from tqdm import tqdm

capture = cv2.VideoCapture("./1_1.mp4")
dir_path = "./images"
os.makedirs(dir_path, exist_ok=True)

for n in tqdm(range(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))):
    ret, frame = capture.read()
    # 読み込めなかった場合はループを抜ける
    if not ret:
        break

    if int(n % 6) == 0:
        cv2.imwrite(f"./{dir_path}/{int(n // 6)}.png", frame)
