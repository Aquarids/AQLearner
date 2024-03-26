import unittest
import numpy as np

from ml.linear.linear_regression import LinearRegression
from ml.linear.logistic_regression import LogisticRegression
from ml.cluster.k_means import KMeans
from ml.cluster.db_scan import DBScan
from ml.tree.id3 import ID3
from ml.tree.c45 import C45
from ml.tree.cart import CART

import sklearn.datasets as datasets
import sklearn.model_selection
import matplotlib.pyplot as plt

class TestLinear(unittest.TestCase):
    def test_linear_regression(self):
        X, y = datasets.make_regression(n_samples=20, n_features=1, noise=0.1)

        train_X, train_y = X[:18], y[:18]
        test_X, test_y = X[18:], y[18:]
        
        model = LinearRegression()
        model.fit(train_X, train_y)

        y_pred = model.predict(test_X)
        print('R-squared:', model.r_squared_score(test_y, y_pred))

    def test_logistic_regression(self):
        X, y = datasets.make_classification(n_samples=20, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

        train_X, train_y = X[:18], y[:18]
        test_X, test_y = X[18:], y[18:]
        
        model = LogisticRegression()
        model.fit(train_X, train_y)

        y_pred = model.predict(test_X)
        print('Logistic Regression Accuracy:', model.accuracy_score(test_y, y_pred))

class TestCluster(unittest.TestCase):
    def test_kmeans(self):
        X, _ = datasets.make_blobs(n_samples=10, centers=3, cluster_std=0.60, random_state=0)
    
        model = KMeans(n_clusters=3)
        model.fit(X)
        labels = model.get_labels()

        plt.title('KMeans Clustering')
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
        centroids = model.centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.5)
        plt.show()

    def test_dbscan(self):
        X, _ = datasets.make_moons(n_samples=100, noise=0.1, random_state=30)
        eps = 0.25
        min_samples = 5

        model = DBScan(eps=eps, min_samples=min_samples)
        model.fit(X)
        labels = model.get_labels()

        plt.figure(figsize=(10, 6))
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -2:  # Noise points
                color = 'k'
            else:
                color = plt.cm.Spectral(float(label) / len(unique_labels))
            class_member_mask = (labels == label)
            xy = X[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=color, markeredgecolor='k', markersize=14 if label == -2 else 6)

        plt.title('DBSCAN Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

class TestTree(unittest.TestCase):
    def test_id3(self):
        X, y = datasets.load_iris(return_X_y=True)
        train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = ID3()
        model.fit(train_X, train_y)

        y_pred = model.predict(test_X)
        print('ID3 Tree:', model.get_tree())
        print('ID3 Accuracy:', model.accuracy_score(test_y, y_pred))

    def test_c45(self):
        X, y = datasets.load_iris(return_X_y=True)
        train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = C45()
        model.fit(train_X, train_y)

        y_pred = model.predict(test_X)
        print('C45 Tree:', model.get_tree())
        print('C45 Accuracy:', model.accuracy_score(test_y, y_pred))

    def test_cart(self):
        X, y = datasets.load_iris(return_X_y=True)
        train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = CART()
        model.fit(train_X, train_y)

        y_pred = model.predict(test_X)
        print('CART Tree:', model.get_tree())
        print('CART Accuracy:', model.accuracy_score(test_y, y_pred))

if __name__ == '__main__':
    unittest.main()