from fl.client import Client
from fl.server import Server
from fl.controller import FLController
from fl.model_factory import ModelFactory
from fl.model_factory import type_regression, type_binary_classification, type_multi_classification
from fla.defend.robust_aggr.robust_aggr_server import RobustAggrServer, type_aggr_median, type_aggr_trimmed_mean
from fla.defend.robust_aggr.robust_aggr_controller import MedianAggrFLController, TrimmedMeanAggrFLController
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

    def train(self, train_loader, test_loader, name):
        print(f"{name} Training")
        model = SimpleCNNClassifier()
        model.fit(train_loader, n_iters=1)
        y_pred, _ = model.predict(test_loader)

        y_test = []
        for _, y in test_loader:
            y_test += y.tolist()
        accuracy = Metrics.accuracy(np.array(y_test), np.array(y_pred))
        return accuracy

    def test_data_poison(self):
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
        label_flip_loader = data_poison.label_flip(train_loader, flip_ratio=0.7, num_classes=10)
        sample_poisoned_loader = data_poison.sample_poison(train_loader, poison_ratio=0.7, noise_level=10)
        ood_data_loader = data_poison.ood_data(train_loader, poison_ratio=0.7)

        normal_accuracy = self.train(train_loader, test_loader, "Normal")
        label_flip_accuracy = self.train(label_flip_loader, test_loader, "Label Flip")
        sample_poisoned_accuracy = self.train(sample_poisoned_loader, test_loader, "Sample Poison")
        ood_data_accuracy = self.train(ood_data_loader, test_loader, "OOD Data")

        print("Normal accuracy: ", normal_accuracy)
        print("Label Flip accuracy: ", label_flip_accuracy)
        print("Sample Poison accuracy: ", sample_poisoned_accuracy)
        print("OOD Data accuracy: ", ood_data_accuracy)

class TestRobustAggr(unittest.TestCase):
    def init_clients(self, n_clients, model_factory_json):
        clients = []
        for _ in range(n_clients):
            model, _, optimizer, criterion = ModelFactory().create_model(model_factory_json)
            client = Client(model, criterion, optimizer)
            clients.append(client)
        return clients
    
    def init_server(self, model_factory_json):
        model, model_type, optimizer, criterion = ModelFactory().create_model(model_factory_json)
        return RobustAggrServer(model, optimizer, criterion, model_type)
    
    def model_factory_json(self):
        return {
            "model_type": type_multi_classification,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "criterion": "cross_entropy",
            "layers": [
                {
                    "type": "conv2d",
                    "in_channels": 1,
                    "out_channels": 32,
                    "kernel_size": 3,
                    "padding": 1,
                    "activation": "relu",
                    "stride": 1
                },
                {
                    "type": "maxpool",
                    "kernel_size": 2,
                    "stride": 2,
                    "padding": 0
                },
                {
                    "type": "conv2d",
                    "in_channels": 32,
                    "out_channels": 64,
                    "kernel_size": 3,
                    "padding": 1,
                    "activation": "relu",
                    "stride": 1
                },
                {
                    "type": "maxpool",
                    "kernel_size": 2,
                    "stride": 2,
                    "padding": 0
                },
                {
                    "type": "reshape",
                    "shape": [-1, 64 * 7 * 7]
                },
                {
                    "type": "linear",
                    "in_features": 7 * 7 * 64,
                    "out_features": 128
                },
                {
                    "type": "relu"
                },
                {
                    "type": "linear",
                    "in_features": 128,
                    "out_features": 10
                },
                {
                    "type": "softmax",
                    "dim": 1
                }
            ]
        }
    
    def normal_compare(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                                download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        data_poison = DataPoison()
        sample_poisoned_loader = data_poison.sample_poison(train_loader, poison_ratio=0.7, noise_level=10)

        n_clients = 6
        n_poisoned_clients = 2 # assume poisoned client less than normal clients (1/3)
        n_rounds = 2
        n_iter = 1

        clients = self.init_clients(n_clients, self.model_factory_json())
        model, model_type, optimizer, criterion = ModelFactory().create_model(self.model_factory_json())
        server = Server(model, optimizer, criterion, model_type)
        controller = FLController(server, clients)

        for i in range(n_clients):
            if i < n_poisoned_clients:
                clients[i].setDataLoader(sample_poisoned_loader, n_iter)
            else:
                clients[i].setDataLoader(train_loader, n_iter)
        server.setTestLoader(test_loader)

        controller.train(n_rounds)
        server.model_metric.summary()

    def test_median_aggr(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                                download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        data_poison = DataPoison()
        sample_poisoned_loader = data_poison.sample_poison(train_loader, poison_ratio=0.7, noise_level=10)

        n_clients = 6
        n_poisoned_clients = 2 # assume poisoned client less than normal clients (1/3)
        n_rounds = 2
        n_iter = 1

        clients = self.init_clients(n_clients, self.model_factory_json())
        server = self.init_server(self.model_factory_json())
        controller = MedianAggrFLController(server, clients)

        for i in range(n_clients):
            if i < n_poisoned_clients:
                clients[i].setDataLoader(sample_poisoned_loader, n_iter)
            else:
                clients[i].setDataLoader(train_loader, n_iter)
        server.setTestLoader(test_loader)

        controller.train(n_rounds)
        server.model_metric.summary()

    def test_trimmed_mean_aggr(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                                download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        data_poison = DataPoison()
        sample_poisoned_loader = data_poison.sample_poison(train_loader, poison_ratio=0.7, noise_level=10)

        n_clients = 6
        n_poisoned_clients = 2 # assume poisoned client less than normal clients (1/3)
        n_rounds = 2
        n_iter = 1

        clients = self.init_clients(n_clients, self.model_factory_json())
        server = self.init_server(self.model_factory_json())
        controller = TrimmedMeanAggrFLController(server, clients, trim_ratio=0.2)

        for i in range(n_clients):
            if i < n_poisoned_clients:
                clients[i].setDataLoader(sample_poisoned_loader, n_iter)
            else:
                clients[i].setDataLoader(train_loader, n_iter)
        server.setTestLoader(test_loader)

        controller.train(n_rounds)
        server.model_metric.summary()