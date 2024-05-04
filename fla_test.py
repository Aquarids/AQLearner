from fl.client import Client
from fl.server import Server
from fl.model_factory import ModelFactory
from dl.simple_cnn_classifier import SimpleCNNClassifier
from dl.simple_logistic_regression import SimpleLogisticRegression
import dl.metrics as Metrics

from fla.data_poison import DataPoison

import numpy as np
import torch
import torch.utils.data
import torchvision
import sklearn.datasets
import sklearn.model_selection
import unittest

class TestDataPoison(unittest.TestCase):
    def test_binary_label_flip(self):
        X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=42)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=10, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=10, shuffle=False)

        data_poison = DataPoison()
        poisoned_loader = data_poison.binary_label_flip(train_loader)
        num_features = X_train.shape[1]

        print("Normal training")
        model = SimpleLogisticRegression(num_features, 1)
        model.fit(train_loader, n_iters=10)
        y_pred, _ = model.predict(test_loader)

        y_test = []
        for _, y in test_loader:
            y_test += y.tolist()
        normal_accuracy = Metrics.accuracy(np.array(y_test), np.array(y_pred))

        print("Poison training")
        poisoned_model = SimpleLogisticRegression(num_features, 1)
        poisoned_model.fit(poisoned_loader, n_iters=10)
        poisoned_y_pred, _ = poisoned_model.predict(test_loader)

        poisoned_accuracy = Metrics.accuracy(np.array(y_test), np.array(poisoned_y_pred))

        print("Normal accuracy: ", normal_accuracy)
        print("Poison accuracy: ", poisoned_accuracy)
        print("Diff accuracy: ", normal_accuracy - poisoned_accuracy)

    def test_sample_poison(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                                download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        data_poison = DataPoison()
        poisoned_loader = data_poison.sample_poison(train_loader, poison_ratio=0.5, noise_level=3)

        print("Normal training")
        model = SimpleCNNClassifier()
        model.fit(train_loader, n_iters=1)
        y_pred, _ = model.predict(test_loader)

        y_test = []
        for _, y in test_loader:
            y_test += y.tolist()
        normal_accuracy = Metrics.accuracy(np.array(y_test), np.array(y_pred))

        print("Poison training")
 
        poisoned_model = SimpleCNNClassifier()
        poisoned_model.fit(poisoned_loader, n_iters=1)
        poisoned_y_pred, _ = poisoned_model.predict(test_loader)

        poisoned_accuracy = Metrics.accuracy(np.array(y_test), np.array(poisoned_y_pred))

        print("Normal accuracy: ", normal_accuracy)
        print("Poison accuracy: ", poisoned_accuracy)
        print("Diff accuracy: ", normal_accuracy - poisoned_accuracy)

    def test_ood_data(unittest):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                                download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        data_poison = DataPoison()
        poisoned_loader = data_poison.ood_data(train_loader, poison_ratio=0.7)

        print("Normal training")
        model = SimpleCNNClassifier()
        model.fit(train_loader, n_iters=1)
        y_pred, _ = model.predict(test_loader)

        y_test = []
        for _, y in test_loader:
            y_test += y.tolist()
        normal_accuracy = Metrics.accuracy(np.array(y_test), np.array(y_pred))

        print("Poison training")
 
        poisoned_model = SimpleCNNClassifier()
        poisoned_model.fit(poisoned_loader, n_iters=1)
        poisoned_y_pred, _ = poisoned_model.predict(test_loader)

        poisoned_accuracy = Metrics.accuracy(np.array(y_test), np.array(poisoned_y_pred))

        print("Normal accuracy: ", normal_accuracy)
        print("Poison accuracy: ", poisoned_accuracy)
        print("Diff accuracy: ", normal_accuracy - poisoned_accuracy)