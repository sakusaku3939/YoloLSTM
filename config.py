import torch.nn as nn
import torch.optim as optim

from helper.validation_functions import get_r2_accuracy
from datasets.dataset_utils import load_cropped_image, load_image, load_test_image, load_cropped_test_image
from models.base.GoogLeNet import calc_loss

c = {
    "general": {
        "num_epochs": 10,
        "random_state": 111,
        "batch_size": 5,
        "num_workers": 2,
        "device": "cuda",
        "checkpoint_resume": False,
    },
    "wandb": {
        "state": False,
        "project": "ImageBasedLocalization_Regress",
        "config": {}
    },
    "models": {
        "YoloLSTM": {
            "state": True,
            "name": "YoloLSTM",
            "train_settings": {
                "data_loader_function": (load_cropped_image, load_cropped_test_image),
                "loss_function": nn.MSELoss(),
                "optimizer": optim.Adam,
                "eval_function": get_r2_accuracy,
            },
            "param": {},
        },
        "PoseLSTM": {
            "state": False,
            "name": "PoseLSTM",
            "train_settings": {
                "data_loader_function": (load_image, load_test_image),
                "loss_function": calc_loss,
                "optimizer": optim.Adam,
                "eval_function": get_r2_accuracy,
            },
            "param": {},
        },
        "PoseNet": {
            "state": False,
            "name": "PoseNet",
            "train_settings": {
                "data_loader_function": (load_image, load_test_image),
                "loss_function": calc_loss,
                "optimizer": optim.Adam,
                "eval_function": get_r2_accuracy,
            },
            "param": {},
        },
        "SimpleCNN": {
            "state": False,
            "name": "SimpleCNN",
            "train_settings": {
                "data_loader_function": (load_image, load_test_image),
                "loss_function": nn.MSELoss(),
                "optimizer": optim.Adam,
                "eval_function": get_r2_accuracy,
            },
            "param": {},
        },
    },
}


def get_config(*keys):
    config = c
    for key in keys:
        config = config[key]
    return config
