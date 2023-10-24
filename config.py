import torch.nn as nn
import torch.optim as optim

from models.validation_functions import get_r2_accuracy

c = {
    "general": {
        "num_epochs": 20,
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
                "loss_function": nn.MSELoss(),
                "optimizer": optim.Adam,
                "eval_function": get_r2_accuracy,
            },
            "param": {},
        },
        "SimpleCNN": {
            "name": "SimpleCNN",
            "state": True,
            "checkpoint_resume": False,
            "train_settings": {
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
            "epochs": 12,
        }
    },
}


def get_config(*keys):
    config = c
    for key in keys:
        config = config[key]
    return config
