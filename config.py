import torch.nn as nn
import torch.optim as optim

from models.validation_functions import get_classification_accuracy

c = {
    "general": {
        "num_epochs": 10,
        "random_state": 111,
        "batch_size": 5,
        "num_workers": 2,
        "device": "cuda",
    },
    "data": {

    },
    "models": {
        "CNNLSTM": {
            "name": "CNNLSTM",
            "state": True,
            "train_settings": {
                "loss_function": nn.CrossEntropyLoss(),
                "optimizer": optim.Adam,
                "eval_function": get_classification_accuracy,
            },
            "param": {},
        },
    },
    "wandb": {
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
}


def get_config(*keys):
    config = c
    for key in keys:
        config = config[key]
    return config
