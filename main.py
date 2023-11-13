import torch
import matplotlib.pyplot as plt
from torch import nn

from tqdm import tqdm
import os
from datetime import datetime
import wandb
import random
import numpy as np
import ssl

from helper.confusion_matrix import show_confusion_matrix
from helper.model_utils import get_models

from config import get_config

ssl._create_default_https_context = ssl._create_unverified_context

# 乱数シードを固定
random_state = get_config("general", "random_state")
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)
torch.set_printoptions(sci_mode=False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


# GPUデバイスの設定
def init_device(config_gen):
    torch.multiprocessing.freeze_support()
    if config_gen["device"] == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config_gen["device"])
    print(f"Device type: {device}")
    return device


def train():
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(f"outputs/{now}")

    config_gen = get_config("general")
    device = init_device(config_gen)

    num_epochs = config_gen["num_epochs"]
    batch_size = config_gen["batch_size"]
    num_workers = config_gen["num_workers"]

    config_wandb = get_config("wandb")
    save_epoch, save_loss = 0, 0.0

    for model, config in get_models():
        os.makedirs(f"outputs/{now}/" + config["name"])
        wandb.init(project=config_wandb["project"], config=config_wandb["config"],
                   mode="online" if config_wandb["state"] else "disabled", resume=config["checkpoint_resume"])
        model = model.to(device)

        train_settings = config["train_settings"]
        loss_function = train_settings["loss_function"]
        optimizer = train_settings["optimizer"](model.parameters())
        train_loader, valid_loader = train_settings["data_loader_function"][0](batch_size, num_workers, random_state)
        results = ""

        # チェックポイントから学習を再開
        checkpoint_epoch = 1
        if config["checkpoint_resume"]:
            path = "outputs\\20230808161130\\YoloLSTM\\training_state.pt"
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Optimizerのstateを現在のdeviceに移す
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            checkpoint_epoch = checkpoint['epoch']

        for epoch in range(checkpoint_epoch, num_epochs + 1):
            model = model.train()
            running_loss = 0.0
            print(f"Epoch: {epoch}")

            for i, data in tqdm(enumerate(train_loader, 0)):
                inputs = [d.to(device) for d in data[0]] if type(data[0]) is list else data[0].to(device)
                labels = data[1].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # 各エポック後の検証
            with torch.no_grad():
                model = model.eval()
                running_score = 0.0

                for j, data in tqdm(enumerate(valid_loader, 0)):
                    inputs = [d.to(device) for d in data[0]] if type(data[0]) is list else data[0].to(device)
                    labels = data[1].to(device)

                    pred = model(inputs)
                    running_score += config["train_settings"]["eval_function"](pred, labels)

            epoch_loss, epoch_score = running_loss / (i + 1), running_score / (j + 1)
            wandb.log({"Epoch": epoch, "Loss": epoch_loss, "Score": epoch_score})
            save_epoch, save_loss = epoch, running_loss

            result = f"Loss: {epoch_loss}  Score: {epoch_score}\n"
            results += ("Epoch:" + str(epoch) + "  " + f"Loss: {epoch_loss}  Score: {epoch_score}\n")
            print(result)

        # モデル学習完了後の保存処理
        out_dir = f"outputs/{now}/" + config["name"] + "/"
        torch.save(model.state_dict(), out_dir + "model.pth")
        if config_wandb["state"]:
            torch.save({'epoch': save_epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': save_loss},
                       out_dir + "training_state.pt")
        with open(out_dir + "results.txt", "w") as file:
            file.write(results)
        wandb.finish()
        print("Training finished")


def predict():
    torch.multiprocessing.freeze_support()
    config_gen = get_config("general")
    batch_size = config_gen["batch_size"]
    num_workers = config_gen["num_workers"]
    device = init_device(config_gen)

    classes = ["(0, 0)", "(1, 11)", "(9, 0)", "(9, 12)", "(13, 0)", "(13, 12)"]

    for model, config in get_models():
        class_correct = list(0. for _ in range(len(classes)))
        class_total = list(0. for _ in range(len(classes)))
        y_pred = []
        y_true = []

        path = "output\\" + config["name"] + "\\model.pth"
        model = model.to(device)
        model = model.eval()
        model.load_state_dict(torch.load(path))

        train_settings = config["train_settings"]
        test_loader = train_settings["data_loader_function"][1](batch_size, num_workers)

        with torch.no_grad():
            for data in test_loader:
                inputs = [d.to(device) for d in data[0]] if type(data[0]) is list else data[0].to(device)
                labels = data[1].to(device)

                # 正解数をカウントする
                pred = model(inputs)
                _, pred = torch.max(pred.data, dim=1)

                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += (pred == labels).sum().item()
                    class_total[label] += test_loader.batch_size
                    y_pred.append(pred[i].item())
                    y_true.append(label.item())

                # 分類に失敗した画像を表示する
                if (pred == labels).sum().item() != labels.size(0):
                    for i in range(len(labels)):
                        if pred[i] != labels[i]:
                            images, labels = data
                            # show_img(torchvision.utils.make_grid(images[i]))
                            print(f'Predicted: {classes[pred[i]]}, Label: {classes[labels[i]]}')

        show_confusion_matrix(y_pred, y_true, classes, config["name"])
        print()
        for i in range(len(classes)):
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


# 画像の表示関数
def show_img(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    # train()
    predict()
