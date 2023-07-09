import torch
import torchvision
import matplotlib.pyplot as plt

from tqdm import tqdm
import os
from datetime import datetime
import wandb
import random
import numpy as np
import ssl

from helper.dataset_utils import load_image, load_test_image
from helper.model_utils import get_models

from config import get_config

ssl._create_default_https_context = ssl._create_unverified_context

random_state = get_config("general", "random_state")
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)


# GPUデバイスの設定
def init_device(config_gen):
    torch.multiprocessing.freeze_support()
    if config_gen["device"] == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config_gen["device"])
    return device


def train():
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(f"outputs/{now}")

    config_gen = get_config("general")
    device = init_device(config_gen)
    train_loader, test_loader = load_image()

    num_epochs = config_gen["num_epochs"]

    for model, config in get_models():
        os.makedirs(f"outputs/{now}/" + config["name"])
        wandb.init(project="ImageBasedLocalization_" + config["name"], config=get_config("wandb"))
        model = model.to(device)

        train_settings = config["train_settings"]
        loss_function = train_settings["loss_function"]
        optimizer = train_settings["optimizer"](model.parameters())
        results = ""

        for epoch in range(num_epochs):
            model = model.train()
            running_loss = 0.0
            i = 0
            print(f"Epoch: {epoch + 1}")

            for i, data in tqdm(enumerate(train_loader, 0)):
                inputs, labels = [d.to(device) for d in data[0]], data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                print(outputs.shape)
                print(labels.shape)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # 各エポック後の処理
            with torch.no_grad():
                model = model.eval()
                running_score = 0.0

                for j, data in tqdm(enumerate(test_loader, 0)):
                    inputs, labels = [d.to(device) for d in data[0]], data[1].to(device)
                    pred = model(inputs)
                    running_score += config["train_settings"]["eval_function"](pred, labels)

            epoch_loss, epoch_score = running_loss / (i + 1), running_score / (j + 1)
            wandb.log({"Loss": epoch_loss, "Score": epoch_score})
            result = f"Loss: {epoch_loss}  Score: {epoch_score}\n"
            results += ("Epoch:" + str(epoch + 1) + "  " + f"Loss: {epoch_loss}  Score: {epoch_score}\n")
            print(result)

        # モデル学習完了後の処理
        out_dir = f"outputs/{now}/" + config["name"] + "/"
        torch.save(model.state_dict(), out_dir + "model.pth")
        with open(out_dir + "results.txt", "w") as file:
            file.write(results)
        wandb.finish()
        print("Training finished")


def predict():
    torch.multiprocessing.freeze_support()
    config_gen = get_config("general")
    device = init_device(config_gen)

    test_loader = load_test_image()
    path = "outputs\\20230628221102\\SimpleCNN\\model.pth"

    classes = ("0_0", "13_12")
    class_correct = list(0. for _ in range(2))
    class_total = list(0. for _ in range(2))

    for model, config in get_models():
        model = model.to(device)
        model = model.eval()
        model.load_state_dict(torch.load(path))

        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)

                # 正解数をカウントする
                _, pred = torch.max(outputs, 1)
                for i in range(2):
                    label = labels[i]
                    class_correct[label] += (pred == labels).sum().item()
                    class_total[label] += test_loader.batch_size

                # 分類に失敗した画像を表示する
                if (pred == labels).sum().item() != labels.size(0):
                    for i in range(0, len(labels)):
                        if pred[i] != labels[i]:
                            images, labels = data
                            show_img(torchvision.utils.make_grid(images[i]))
                            print(f'Predicted: {classes[pred[i]]}, Label: {classes[labels[i]]}')

    print()
    for i in range(2):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


# 画像の表示関数
def show_img(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    train()
    # predict()
