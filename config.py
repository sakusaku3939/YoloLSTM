import torch.nn as nn
import torch.optim as optim

from helper.dataset_utils import load_cropped_image, load_image
from models.GoogLeNet import calc_loss
from models.validation_functions import get_classification_accuracy

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
                "data_loader_function": load_cropped_image,
                "loss_function": nn.CrossEntropyLoss(),
                "optimizer": optim.Adam,
                "eval_function": get_classification_accuracy,
            },
            "param": {},
        },
        "CNNLSTM": {
            "name": "CNNLSTM",
            "state": False,
            "checkpoint_resume": False,
            "train_settings": {
                "data_loader_function": load_image,
                "loss_function": calc_loss,
                "optimizer": optim.Adam,
                "eval_function": get_classification_accuracy,
            },
            "param": {},
        },
        "GoogLeNet": {
            "name": "GoogLeNet",
            "state": False,
            "checkpoint_resume": False,
            "train_settings": {
                "data_loader_function": load_image,
                "loss_function": calc_loss,
                "optimizer": optim.Adam,
                "eval_function": get_classification_accuracy,
            },
            "param": {
                "num_classes": 3
            },
        },
        "SimpleCNN": {
            "name": "SimpleCNN",
            "state": False,
            "checkpoint_resume": False,
            "train_settings": {
                "data_loader_function": load_image,
                "loss_function": nn.CrossEntropyLoss(),
                "optimizer": optim.Adam,
                "eval_function": get_classification_accuracy,
            },
            "param": {},
        },
    },
    "wandb": {
        "state": True,
        "project": "ImageBasedLocalization_Classify",
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
