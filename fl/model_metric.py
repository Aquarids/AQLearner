import dl.metrics as Metrics
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from fl.model_factory import type_regression, type_binary_classification, type_multi_classification


class ModelMetric:

    def __init__(self, type) -> None:
        if type != type_regression and type != type_multi_classification and type != type_binary_classification:
            raise ValueError("Invalid type of model metric")
        self.type = type
        self.reset()

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
        y_true, y_pred = np.array(y_true).astype(float), np.array(
            y_pred).astype(float)
        if y_prob is not None:
            y_prob = np.array(y_prob).astype(float)

        self.y_true_list.append(y_true)
        self.y_pred_list.append(y_pred)
        if y_prob is not None:
            self.y_prob_list.append(y_prob)
        self.caculate(y_true, y_pred, y_prob, id_round)

    def caculate(self, y_true, y_pred, y_prob, id_round):
        if self.type == type_regression:
            self.mse.append(Metrics.mse(y_true, y_pred))
        elif self.type == type_binary_classification:
            self.accuracy.append(Metrics.accuracy(y_true, y_pred))

            precession, recall, f1 = Metrics.precision_recall_f1(
                y_true, y_pred)
            self.precision.append(precession)
            self.recall.append(recall)
            self.f1.append(f1)

            tpr, fpr = Metrics.tpr_fpr(y_true, y_pred)
            self.tpr.append(tpr)
            self.fpr.append(fpr)

            if y_prob is None:
                return
            else:
                self.auc.append(Metrics.auc(y_true, y_prob))

                cur_time = time.time()
                filename = "roc_curve_" + str(id_round) + "_" + str(
                    cur_time) + ".png"
                Metrics.save_roc_curve(y_true, y_prob, filename,
                                       "./fl/metric/plot")
        elif self.type == type_multi_classification:
            self.accuracy.append(Metrics.accuracy(y_true, y_pred))

            precession, recall, f1 = Metrics.precision_recall_f1(
                y_true, y_pred)
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

    def concatenate_lists(self):
        y_true = torch.cat(self.y_true_list, dim=1).numpy()
        y_pred = torch.cat(self.y_pred_list, dim=1).numpy()
        y_prob = torch.cat(self.y_prob_list, dim=1).numpy()
        return y_true, y_pred, y_prob

    def plot_ks_curve(self, filename, dir):
        y_true, y_pred, y_prob = self.concatenate_lists()

        # Number of classes
        num_classes = y_prob.shape[2]

        for i in range(num_classes):
            y_true_binary = (y_true == i).astype(int)
            y_prob_binary = y_prob[:, :, i].flatten()

            # Sort by predicted probabilities
            sorted_indices = np.argsort(y_prob_binary)
            y_true_sorted = y_true_binary.flatten()[sorted_indices]
            y_prob_sorted = y_prob_binary[sorted_indices]

            # Compute the cumulative distributions
            cdf_pos = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
            cdf_neg = np.cumsum(1 - y_true_sorted) / np.sum(1 - y_true_sorted)

            # Calculate the KS statistic
            ks_stat = np.max(np.abs(cdf_pos - cdf_neg))
            ks_index = np.argmax(np.abs(cdf_pos - cdf_neg))
            ks_threshold = y_prob_sorted[ks_index]

            # Plot the KS curve
            plt.figure(figsize=(10, 6))
            plt.plot(y_prob_sorted, cdf_pos, label='Cumulative True Positive Rate', color='blue')
            plt.plot(y_prob_sorted, cdf_neg, label='Cumulative False Positive Rate', color='red')
            plt.axvline(ks_threshold, color='green', linestyle='--', label=f'KS Statistic = {ks_stat:.4f} at threshold {ks_threshold:.4f}')
            plt.xlabel('Predicted Probability')
            plt.ylabel('Cumulative Distribution')
            plt.title(f'KS Curve for Class {i}')
            plt.legend()
            plt.show()

    def plot_roc_curve(self, filename, dir):
        y_true, y_pred, y_prob = self.concatenate_lists()

        # Number of classes
        num_classes = y_prob.shape[2]

        for i in range(num_classes):
            y_true_binary = (y_true == i).astype(int)
            y_prob_binary = y_prob[:, :, i].flatten()

            # Sort by predicted probabilities
            sorted_indices = np.argsort(y_prob_binary)
            y_true_sorted = y_true_binary.flatten()[sorted_indices]
            y_prob_sorted = y_prob_binary[sorted_indices]

            # Compute the ROC curve
            thresholds = np.linspace(0, 1, 100)
            tpr = []
            fpr = []

            for threshold in thresholds:
                tp = np.sum((y_prob_sorted >= threshold) & (y_true_sorted == 1))
                fp = np.sum((y_prob_sorted >= threshold) & (y_true_sorted == 0))
                fn = np.sum((y_prob_sorted < threshold) & (y_true_sorted == 1))
                tn = np.sum((y_prob_sorted < threshold) & (y_true_sorted == 0))

                tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

            plt.plot(fpr, tpr, label=f'Class {i} ROC curve')

        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    def plot_precision_recall_curve(self, filename, dir):
        y_true, y_pred, y_prob = self.concatenate_lists()

        # Number of classes
        num_classes = y_prob.shape[2]

        for i in range(num_classes):
            y_true_binary = (y_true == i).astype(int)
            y_prob_binary = y_prob[:, :, i].flatten()

            # Sort by predicted probabilities
            sorted_indices = np.argsort(y_prob_binary)
            y_true_sorted = y_true_binary.flatten()[sorted_indices]
            y_prob_sorted = y_prob_binary[sorted_indices]

            # Compute Precision-Recall curve
            thresholds = np.linspace(0, 1, 100)
            precision = []
            recall = []

            for threshold in thresholds:
                tp = np.sum((y_prob_sorted >= threshold) & (y_true_sorted == 1))
                fp = np.sum((y_prob_sorted >= threshold) & (y_true_sorted == 0))
                fn = np.sum((y_prob_sorted < threshold) & (y_true_sorted == 1))

                precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
                recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

            plt.plot(recall, precision, label=f'Class {i} Precision-Recall curve')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()

    def summary(self):
        print(torch.tensor(self.y_pred_list).shape)
        print(torch.tensor(self.y_true_list).shape)
        print(torch.tensor(self.y_prob_list).shape)
        cur_time = time.time()
        if self.type == type_regression:
            print("MSE:", self.mse)
            self.save_mse_curve("mse_curve_" + str(cur_time) + ".png",
                                "./fl/metric/plot")
        elif self.type == type_binary_classification:
            print("Accuracy:", self.accuracy)
            print("Precision:", self.precision)
            print("Recall:", self.recall)
            print("F1:", self.f1)
            print("TPR:", self.tpr)
            print("FPR:", self.fpr)
            print("AUC:", self.auc)
            self.save_accuracy_curve(
                "accuracy_curve_" + str(cur_time) + ".png", "./fl/metric/plot")
            self.plot_roc_curve("roc_curve_" + str(cur_time) + ".png",
                               "./fl/metric/plot")
            self.plot_precision_recall_curve(
                "precision_recall_curve_" + str(cur_time) + ".png", "./fl/metric/plot")
            self.plot_ks_curve("ks_curve_" + str(cur_time) + ".png", "./fl/metric/plot")
        elif self.type == type_multi_classification:
            print("Accuracy:", self.accuracy)
            print("Precision:", self.precision)
            print("Recall:", self.recall)
            print("F1:", self.f1)
            print("TPR:", self.tpr)
            print("FPR:", self.fpr)
            self.save_accuracy_curve(
                "accuracy_curve_" + str(cur_time) + ".png", "./fl/metric/plot")
