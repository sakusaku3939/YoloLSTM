import os
import csv

path = "./data/train/0627/"

# X * y = 13 * 12
# 推定で使用する座標を、撮影する順番（右下から左上）に記録
pos_list = [range(1, 8), range(1, 12), range(6, 13), range(6, 8), range(6, 8), range(6, 8), range(6, 8), range(6, 8),
            range(1, 13), range(6, 8), range(6, 8), range(6, 8), range(1, 13)]

file_names = sorted(os.listdir(path))
x = 1
outputs = []

# (image_name, x, y) のラベルデータを作成
for y_list in pos_list:
    for i, y in enumerate(y_list):
        outputs.append([file_names[i], x, y])
    x += 1

# CSVに書き込み
with open(path + "label.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(outputs)
