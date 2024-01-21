import os
import csv

from tools.pos_list import pos_list

path = "./data/0627"

file_names = sorted(os.listdir(path))
x = 1
outputs = []

# (image_name, x, y) のラベルデータを作成
for y_list in pos_list:
    for i, y in enumerate(y_list):
        outputs.append([file_names[i], x, y])
    x += 1

# CSVに書き込み
with open(path + "/label.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(outputs)
