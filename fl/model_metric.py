import dl.metrics as Metrics
import time
import os
import matplotlib.pyplot as plt
import numpy as np

from fl.model_factory import type_regresion, type_binary_classification, type_multi_classification

class ModelMetric:
    def __init__(self, type) -> None:
        if type != type_regresion and type != type_multi_classification and type != type_binary_classification:
            raise ValueError("Invalid type of model metric")
        self.type = type
        self.y_true_list = []
        self.y_pred_list = []
        self.y_prob_list = []

        self.mse = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1 = []
        self.tpr = []
        self.fpr = []
        self.auc = []


    def reset(self):
        self.y_true_list = []
        self.y_pred_list = []
        self.y_prob_list = []

        self.mse = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1 = []
        self.tpr = []
        self.fpr = []
        self.auc = []

    def update(self, y_true, y_pred, y_prob, id_round):
        self.y_true_list.append(y_true)
        self.y_pred_list.append(y_pred)
        if self.type == type_multi_classification:
            self.y_prob_list.append(y_prob)
        self.caculate(y_true, y_pred, y_prob, id_round)

    def caculate(self, y_true, y_pred, y_prob, id_round):
        y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
        if self.type == type_regresion:
            self.mse.append(Metrics.mse(y_true, y_pred))
        elif self.type == type_binary_classification:
            self.accuracy.append(Metrics.accuracy(y_true, y_pred))

            precession, recall, f1 = Metrics.precision_recall_f1(y_true, y_pred)
            self.precision.append(precession)
            self.recall.append(recall)
            self.f1.append(f1)

            tpr, fpr = Metrics.tpr_fpr(y_true, y_pred)
            self.tpr.append(tpr)
            self.fpr.append(fpr)

            self.auc.append(Metrics.auc(y_true, y_prob))

            cur_time = time.time()
            filename = "roc_curve_" + str(id_round) + "_" + str(cur_time) + ".png"
            Metrics.save_roc_curve(y_true, y_prob, filename, "./fl/metric/plot")
        elif self.type == type_multi_classification:
            self.accuracy.append(Metrics.accuracy(y_true, y_pred))

            precession, recall, f1 = Metrics.precision_recall_f1(y_true, y_pred)
            self.precision.append(precession)
            self.recall.append(recall)
            self.f1.append(f1)

            tpr, fpr = Metrics.tpr_fpr(y_true, y_pred)
            self.tpr.append(tpr)
            self.fpr.append(fpr)

    def save_mse_curve(self, filename, dir):
        plt.plot(self.mse)
        plt.xlabel('Round')
        plt.ylabel('MSE')
        plt.title('MSE Curve')

        if not os.path.exists(dir):
            os.makedirs(dir)
        filepath = os.path.join(dir, filename)
        plt.savefig(filepath)
        plt.close()

    def save_accuracy_curve(self, filename, dir):
        plt.plot(self.accuracy)
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')

        if not os.path.exists(dir):
            os.makedirs(dir)
        filepath = os.path.join(dir, filename)
        plt.savefig(filepath)
        plt.close()
            
    def summary(self):
        cur_time = time.time()
        if self.type == type_regresion:
            print("MSE:", self.mse)
            self.save_mse_curve("mse_curve_" + str(cur_time) + ".png", "./fl/metric/plot")
        elif self.type == type_binary_classification:
            print("Accuracy:", self.accuracy)
            print("Precision:", self.precision)
            print("Recall:", self.recall)
            print("F1:", self.f1)
            print("TPR:", self.tpr)
            print("FPR:", self.fpr)
            print("AUC:", self.auc)
            self.save_accuracy_curve("accuracy_curve_" + str(cur_time) + ".png", "./fl/metric/plot")
        elif self.type == type_multi_classification:
            print("Accuracy:", self.accuracy)
            print("Precision:", self.precision)
            print("Recall:", self.recall)
            print("F1:", self.f1)
            print("TPR:", self.tpr)
            print("FPR:", self.fpr)
