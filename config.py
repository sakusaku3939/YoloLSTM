import torch.nn as nn
import torch.optim as optim

from helper.validation_functions import get_classification_accuracy
from datasets.dataset_utils import load_cropped_image, load_image, load_test_image, load_cropped_test_image
from models.base.GoogLeNet import calc_loss

c = {
    "general": {
        "num_epochs": 30,
        "random_state": 111,
        "batch_size": 5,
        "num_workers": 2,
        "device": "cuda",
        "checkpoint_resume": False,
    },
    "wandb": {
        "state": True,
        "project": "ImageBasedLocalization_Classify",
        "config": {}
    },
    "models": {
        "YoloLSTM": {
            "state": True,
            "name": "YoloLSTM",
            "train_settings": {
                "data_loader_function": (load_cropped_image, load_cropped_test_image),
                "loss_function": nn.CrossEntropyLoss(),
                "optimizer": optim.Adam,
                "eval_function": get_classification_accuracy,
            },
            "param": {},
        },
        "PoseLSTM": {
            "state": True,
            "name": "PoseLSTM",
            "train_settings": {
                "data_loader_function": (load_image, load_test_image),
                "loss_function": calc_loss,
                "optimizer": optim.Adam,
                "eval_function": get_classification_accuracy,
            },
            "param": {},
        },
        "PoseNet": {
            "state": True,
            "name": "PoseNet",
            "train_settings": {
                "data_loader_function": (load_image, load_test_image),
                "loss_function": calc_loss,
                "optimizer": optim.Adam,
                "eval_function": get_classification_accuracy,
            },
            "param": {},
        },
        "SimpleCNN": {
            "state": True,
            "name": "SimpleCNN",
            "train_settings": {
                "data_loader_function": (load_image, load_test_image),
                "loss_function": nn.CrossEntropyLoss(),
                "optimizer": optim.Adam,
                "eval_function": get_classification_accuracy,
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
