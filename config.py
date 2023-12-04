import torch.nn as nn
import torch.optim as optim

from helper.validation_functions import get_r2_accuracy
from datasets.dataset_utils import load_cropped_image, load_image, load_test_image, load_cropped_test_image
from models.GoogLeNet import calc_loss

c = {
    "general": {
        "num_epochs": 30,
        "random_state": 111,
        "batch_size": 5,
        "num_workers": 2,
        "device": "cuda",
    },
    "data": {},
    "models": {
        "YoloLSTM": {
            "name": "YoloLSTM",
            "state": True,
            "checkpoint_resume": False,
            "train_settings": {
                "data_loader_function": (load_cropped_image, load_cropped_test_image),
                "loss_function": nn.MSELoss(),
                "optimizer": optim.Adam,
                "eval_function": get_r2_accuracy,
            },
            "param": {},
        },
        "CNNLSTM": {
            "name": "CNNLSTM",
            "state": True,
            "checkpoint_resume": False,
            "train_settings": {
                "data_loader_function": (load_image, load_test_image),
                "loss_function": calc_loss,
                "optimizer": optim.Adam,
                "eval_function": get_r2_accuracy,
            },
            "param": {},
        },
        "GoogLeNet": {
            "name": "GoogLeNet",
            "state": True,
            "checkpoint_resume": False,
            "train_settings": {
                "data_loader_function": (load_image, load_test_image),
                "loss_function": calc_loss,
                "optimizer": optim.Adam,
                "eval_function": get_r2_accuracy,
            },
            "param": {
                "num_classes": 2
            },
        },
        "SimpleCNN": {
            "name": "SimpleCNN",
            "state": True,
            "checkpoint_resume": False,
            "train_settings": {
                "data_loader_function": (load_image, load_test_image),
                "loss_function": nn.MSELoss(),
                "optimizer": optim.Adam,
                "eval_function": get_r2_accuracy,
            },
            "param": {},
        },
    },
    "wandb": {
        "state": True,
        "project": "ImageBasedLocalization_Regress",
        "config": {
            "learning_rate": 0.02,
        }
    },
}


def get_config(*keys):
    config = c
    for key in keys:
        config = config[key]
    return config
