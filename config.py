import torch.nn as nn
import torch.optim as optim

from models.validation_functions import get_classification_accuracy

c = {
    "general": {
        "num_epochs": 2,
        "random_state": 111,
        "batch_size": 10,
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
        "state": False,
        "project": "ImageBasedLocalization_Classify",
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
