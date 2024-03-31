import unittest
import numpy as np
import torch
import dl.metrics as Metrics
import sklearn.datasets
import sklearn.model_selection

from dl.simple_linear_regression import SimpleLinearRegression
from dl.simple_logistic_regression import SimpleLogisticRegression

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
        y_pred = model.predict(X_test)

        y_test_np = y_test.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()

        print('SimpleLogisticRegression Accuracy:', Metrics.accuracy(y_test_np, y_pred_np))



if __name__ == '__main__':
    unittest.main()