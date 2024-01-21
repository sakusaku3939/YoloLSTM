import os

import cv2
from tqdm import tqdm

input_dir = "../inputs"
output_dir = "./images"

for file_name in os.listdir(input_dir):
    video_path = os.path.join(input_dir, file_name)
    capture = cv2.VideoCapture(video_path)

    # 出力フォルダを作成
    video_output_dir = os.path.join(output_dir, file_name.split('.')[0])
    os.makedirs(video_output_dir, exist_ok=True)

    for n in tqdm(range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = capture.read()
        # 読み込めなかった場合はループを抜ける
        if not ret:
            break

        if int(n % 6) == 0:
            frame_file = os.path.join(video_output_dir, f"{int(n // 6)}.png")
            cv2.imwrite(frame_file, frame)
