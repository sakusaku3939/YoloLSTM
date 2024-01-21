import os
import shutil

from tools.pos_list import pos_list

source_path = "./data/0627/"
output_path = "../data/train/"

file_names = iter(sorted(os.listdir(source_path)))

for x, y_list in enumerate(pos_list, start=1):
    for i, y in enumerate(y_list):
        # 座標毎のフォルダを作成
        folder_path = output_path + f"{x}_{y}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 画像ファイルをfolder_pathに移動
        file_name = next(file_names)
        shutil.copy(source_path + file_name, folder_path + "/" + file_name)

