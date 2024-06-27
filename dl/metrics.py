import numpy as np
import matplotlib.pyplot as plt
import os


def accuracy(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.sum(y_test == y_pred) / len(y_test)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


def precision_recall_f1(y_true, y_pred):
    eps = 1e-7
    true_positive = np.sum((y_pred - 1 == 0) & (y_true - 1 == 0))
    false_positive = np.sum((y_pred - 1 == 0) & (y_true - 1 != 0))
    false_negative = np.sum((y_pred - 1 != 0) & (y_true - 1 == 0))
    precision = true_positive / (true_positive + false_positive + eps)
    recall = true_positive / (true_positive + false_negative + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    return precision, recall, f1


def tpr_fpr(y_true, y_pred):
    eps = 1e-7
    true_positive = np.sum((y_pred == 1) & (y_true == 1))
    false_positive = np.sum((y_pred == 1) & (y_true == 0))
    false_negative = np.sum((y_pred == 0) & (y_true == 1))
    true_negative = np.sum((y_pred == 0) & (y_true == 0))
    tpr = true_positive / (true_positive + false_negative + eps)
    fpr = false_positive / (false_positive + true_negative + eps)
    return tpr, fpr


def auc(y_true, y_prob):
    thresholds = np.sort(np.unique(y_prob))[::-1]
    auc = 0
    tpr_prev = 0
    fpr_prev = 0
    for threshold in thresholds:
        y_pred = np.where(y_prob >= threshold, 1, 0)
        tpr, fpr = tpr_fpr(y_true, y_pred)
        auc += (tpr + tpr_prev) / 2 * (fpr - fpr_prev)
        tpr_prev = tpr
        fpr_prev = fpr
    return auc


def plot_roc_curve(y_true, y_prob, show=True):
    thresholds = np.sort(np.unique(y_prob))[::-1]
    tpr_list = []
    fpr_list = []
    for threshold in thresholds:
        y_pred = np.where(y_prob >= threshold, 1, 0)
        tpr, fpr = tpr_fpr(y_true, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    plt.plot(fpr_list, tpr_list)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    if show:
        plt.show()
    return plt


def save_roc_curve(y_true, y_prob, filename, dir):
    plt = plot_roc_curve(y_true, y_prob, show=False)
    if not os.path.exists(dir):
        os.makedirs(dir)
    filepath = os.path.join(dir, filename)
    plt.savefig(filepath)
    plt.close()
    return os.path.abspath(filepath)
