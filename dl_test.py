import unittest
import numpy as np
import torch
import torch.utils.data
import dl.metrics as Metrics
import sklearn.datasets
import sklearn.model_selection
import torchvision.transforms

from dl.simple_linear_regression import SimpleLinearRegression
from dl.simple_logistic_regression import SimpleLogisticRegression
from dl.simple_cnn_classifier import SimpleCNNClassifier
from dl.simple_cnn_regression import SimpleCNNRegression

from sklearn.preprocessing import StandardScaler

class TestNN(unittest.TestCase):
    def test_simple_linear_regression(self):
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=42)

        num_features = X_train.shape[1]
        model = SimpleLinearRegression(num_features, 1)
        model.fit(X_train, y_train)
        model.summary()
        y_pred = model.predict(X_test)

        y_test_np = y_test.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()

        print('SimpleLinearRegression MSE:', Metrics.mse(y_test_np, y_pred_np))

    def test_simple_logistic_regression(self):
        X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=42)

        num_features = X_train.shape[1]
        model = SimpleLogisticRegression(num_features, 1)
        model.fit(X_train, y_train)
        model.summary()
        y_pred = model.predict(X_test)

        y_test_np = y_test.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()

        print('SimpleLogisticRegression Accuracy:', Metrics.accuracy(y_test_np, y_pred_np))

class TestCNN(unittest.TestCase):
    def test_simple_cnn_classifier(self):
        num_samples = 1000
        num_classes = 10

        X = torch.rand((num_samples, 1, 28, 28))  # 1000个28x28的单通道图像
        y_classification = torch.randint(0, num_classes, (num_samples,))
        dataset_classification = torch.utils.data.TensorDataset(X, y_classification)

        # 划分数据集
        train_size_classification = int(0.8 * num_samples)
        test_size_classification = num_samples - train_size_classification
        train_dataset_classification, test_dataset_classification = torch.utils.data.random_split(dataset_classification, [train_size_classification, test_size_classification])

        # 创建DataLoader
        train_loader = torch.utils.data.DataLoader(train_dataset_classification, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset_classification, batch_size=64, shuffle=False)

        model = SimpleCNNClassifier()
        model.fit(train_loader)
        model.summary()
        y_pred, y_possibility = model.predict(test_loader)

        y_test = []
        for _, y in test_loader:
            y_test += y.tolist()
        
        print('SimpleCNNClassifier Accuracy:', Metrics.accuracy(y_test, y_pred))
        print('SimpleCNNClassifier Precision, Recall, F1:', Metrics.precision_recall_f1(y_test, y_pred))
        Metrics.plot_roc_curve(y_test, y_possibility)

    def test_simple_cnn_regression(self):
        num_samples = 1000

        X = torch.rand((num_samples, 1, 28, 28))
        y_regression = torch.randn((num_samples, 1))
        dataset_regression = torch.utils.data.TensorDataset(X, y_regression)

        train_size_regression = int(0.8 * num_samples)
        test_size_regression = num_samples - train_size_regression
        train_dataset_regression, test_dataset_regression = torch.utils.data.random_split(dataset_regression, [train_size_regression, test_size_regression])

        train_loader_regression = torch.utils.data.DataLoader(train_dataset_regression, batch_size=64, shuffle=True)
        test_loader_regression = torch.utils.data.DataLoader(test_dataset_regression, batch_size=64, shuffle=False)

        model = SimpleCNNRegression()
        model.fit(train_loader_regression)
        model.summary()
        y_pred = model.predict(test_loader_regression)

        y_test = []
        for _, y in test_loader_regression:
            y_test += y.tolist()

        print('SimpleCNNRegression MSE:', Metrics.mse(np.array(y_test), np.array(y_pred)))
            

if __name__ == '__main__':
    unittest.main()