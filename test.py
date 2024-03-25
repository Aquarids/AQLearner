import unittest
import numpy as np

from ml.linear.linear_regression import LinearRegression
from ml.linear.logistic_regression import LogisticRegression
from ml.cluster.k_means import KMeans

import sklearn.datasets as datasets
import matplotlib.pyplot as plt

class TestLinearRegression(unittest.TestCase):
    def test_linear_regression(self):
        X, y = datasets.make_regression(n_samples=20, n_features=1, noise=0.1)

        train_X, train_y = X[:18], y[:18]
        test_X, test_y = X[18:], y[18:]
        
        model = LinearRegression()
        model.fit(train_X, train_y)

        y_pred = model.predict(test_X)
        print('R-squared:', model.r_squared_score(test_y, y_pred))

class TestLogisticRegression(unittest.TestCase):
    def test_logistic_regression(self):
        X, y = datasets.make_classification(n_samples=20, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

        train_X, train_y = X[:18], y[:18]
        test_X, test_y = X[18:], y[18:]
        
        model = LogisticRegression()
        model.fit(train_X, train_y)

        y_pred = model.predict(test_X)
        print('Accuracy:', model.accuracy_score(test_y, y_pred))

class TestKMeans(unittest.TestCase):
    def test_kmeans(self):
        X, _ = datasets.make_blobs(n_samples=10, centers=3, cluster_std=0.60, random_state=0)
    
        model = KMeans(n_clusters=3)
        model.fit(X)
        predicted_labels = model.predict(X)

        plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, s=50, cmap='viridis')
        centroids = model.centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.5)
        plt.show()

if __name__ == '__main__':
    unittest.main()