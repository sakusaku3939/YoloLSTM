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

from helper.confusion_matrix import show_confusion_matrix
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
    train_loader, valid_loader = load_image()

    num_epochs = config_gen["num_epochs"]
    config_wandb = get_config("wandb")

    for model, config in get_models():
        os.makedirs(f"outputs/{now}/" + config["name"])
        wandb.init(project=config_wandb["project"], config=config_wandb["config"],
                   mode="online" if config_wandb["state"] else "disabled")
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
                inputs, target = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # 各エポック後の検証
            with torch.no_grad():
                model = model.eval()
                pred_list = []
                target_list = []

                for j, data in tqdm(enumerate(valid_loader, 0)):
                    inputs, target = data[0].to(device), data[1].to(device)
                    pred = model(inputs)
                    pred_list.append(pred)
                    target_list.append(target)

            pred_list = torch.cat(pred_list)
            target_list = torch.cat(target_list)
            running_score = config["train_settings"]["eval_function"](pred_list, target_list)
            epoch_loss = running_loss / (i + 1)

            wandb.log({"Epoch": epoch + 1, "Loss": epoch_loss, "Score": running_score})
            result = f"Loss: {epoch_loss}  Score: {running_score}\n"
            results += ("Epoch:" + str(epoch + 1) + "  " + f"Loss: {epoch_loss}  Score: {running_score}\n")
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
    path = "outputs\\20230808222738\\SimpleCNN\\model.pth"

    for model, config in get_models():
        model = model.to(device)
        model = model.eval()
        model.load_state_dict(torch.load(path))

        train_settings = config["train_settings"]
        loss_function = train_settings["loss_function"]

        with torch.no_grad():
            model = model.eval()
            pred_list = []
            target_list = []
            running_loss = 0.0

            for j, data in tqdm(enumerate(test_loader, 0)):
                inputs, target = data[0].to(device), data[1].to(device)
                pred = model(inputs)
                loss = loss_function(pred, target)
                running_loss += loss.item()

                pred_list.append(pred)
                target_list.append(target)

            pred_list = torch.cat(pred_list)
            target_list = torch.cat(target_list)
            running_score = config["train_settings"]["eval_function"](pred_list, target_list)
            print(f"Loss: {running_loss / (j + 1)}  Score: {running_score}\n")


# 画像の表示関数
def show_img(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    train()
    # predict()
