import numpy as np
import matplotlib.pyplot as plt

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def precision_recall_f1(y_true, y_pred):
    true_positive = np.sum((y_pred == 1) & (y_true == 1))
    false_positive = np.sum((y_pred == 1) & (y_true == 0))
    false_negative = np.sum((y_pred == 0) & (y_true == 1))
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def tpr_fpr(y_true, y_pred):
    true_positive = np.sum((y_pred == 1) & (y_true == 1))
    false_positive = np.sum((y_pred == 1) & (y_true == 0))
    false_negative = np.sum((y_pred == 0) & (y_true == 1))
    true_negative = np.sum((y_pred == 0) & (y_true == 0))
    tpr = true_positive / (true_positive + false_negative)
    fpr = false_positive / (false_positive + true_negative)
    return tpr, fpr

def auc(y_true, y_prob):
    thresholds = np.sort(np.unique(y_prob))[::-1]
    auc = 0
    tpr_prev = 0
    for threshold in thresholds:
        y_pred = np.where(y_prob >= threshold, 1, 0)
        tpr, fpr = tpr_fpr(y_true, y_pred)
        auc += (tpr + tpr_prev) / 2 * (fpr - fpr_prev)
        tpr_prev = tpr
        fpr_prev = fpr
    return auc

def plot_roc_curve(y_true, y_prob):
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
    plt.show()

