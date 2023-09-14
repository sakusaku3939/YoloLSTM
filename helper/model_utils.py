from models.CNNLSTM import CNNLSTM
from models.GoogLeNet import GoogLeNet
from models.YoloLSTM import YoloLSTM
from config import get_config
import sys

sys.path.append('../')

models = {
    "YoloLSTM": YoloLSTM,
    "CNNLSTM": CNNLSTM,
    "GoogLeNet": GoogLeNet,
}


# stateがTrueのモデルとその設定一覧を返す
def get_models():
    selected_models = []
    for key in models:
        model_config = get_config("models")[key]
        if model_config["state"]:
            selected_models.append([models[key](model_config["param"]), model_config])
    return selected_models
