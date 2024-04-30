# YoloLSTM: Image-based Localization model
"Demo: Image-based Indoor Localization using Object Detection and LSTM"  

### Setup
- Python 3.11

- Torch 2.0.1+, Torchvision 0.15.2+ and CUDA 11.7+
```
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

- Installing Packages
```
pip install -r requirements.txt
```
<br>

- Change branch to `i208_regression`. The `master` branch cannot be used.
```
git checkout i208_regression
```

- Place the [i208 laboratory dataset](https://www.dropbox.com/scl/fo/1dx4lj088k04iglkcszpm/ABEHgkJfKdooGy2AsX0mMgU?rlkey=i1mjuag211gde0w9s9m52m50j&st=j9v2p39p&dl=0) in the root `data_all/train`, `data_all/valid` and `data_all/test`.
```
data_all/
├── train/
├── valid/
└── test/
```

- Start learning with the following command.
```
python main.py
```

### Evaluation
You can compare with other models by setting `state` in `config.py` to True/False.
- `PoseNet`: Implementation of [PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.html) [Kendall2015ICCV]
- `PoseLSTM`: Implementation of [Image-Based Localization Using LSTMs for Structured Feature Correlation](https://openaccess.thecvf.com/content_iccv_2017/html/Walch_Image-Based_Localization_Using_ICCV_2017_paper.html) [Walch2017ICCV]
- `SimpleCNN`: CNN with the same number of layers as YoloLSTM using the whole image as input

