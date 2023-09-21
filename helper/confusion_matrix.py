import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def show_confusion_matrix(y_pred, y_true, class_labels, model_name):
    # Confusion Matrixを生成して正規化
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Confusion Matrixを描画
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt=".0f",
                     xticklabels=class_labels, yticklabels=class_labels, vmin=0, vmax=100)
    for t in ax.texts:
        t.set_text(t.get_text() + " %")

    plt.title(model_name)
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.show()
