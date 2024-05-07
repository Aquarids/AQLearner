from fl.client import Client
from fl.server import Server
from fl.controller import FLController, mode_avg_grad, mode_avg_weight, mode_avg_vote
from fl.model_factory import ModelFactory
from fl.model_factory import type_regression, type_binary_classification, type_multi_classification
from fla.malicious_client import MaliciousClient
from fla.malicious_client import attack_type_none, attack_sample_poison, attack_label_flip, attack_ood_data, attack_backdoor, attack_gradient_poison, attack_weight_poison
from fla.defend.robust_aggr.robust_aggr_server import RobustAggrServer
from fla.defend.robust_aggr.robust_aggr_controller import MedianAggrFLController, TrimmedMeanAggrFLController
from dl.simple_cnn_classifier import SimpleCNNClassifier
from dl.simple_logistic_regression import SimpleLogisticRegression
import dl.metrics as Metrics

import numpy as np
import torch
import torch.utils.data
import torchvision
import sklearn.datasets
import sklearn.model_selection
import random
import unittest

class TestDataPoison(unittest.TestCase):
    def test_binary_label_flip(self):
        X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=42)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=10, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=10, shuffle=False)

        num_features = X_train.shape[1]

        y_test = []
        for _, y in test_loader:
            y_test += y.tolist()

        print("Normal training")
        model = SimpleLogisticRegression(num_features, 1)
        normal_client = Client(model, torch.nn.BCELoss(), torch.optim.Adam(model.parameters(), lr=0.01), type=type_binary_classification)
        normal_client.setDataLoader(train_loader, n_iters=1)
        normal_client.train()

        y_pred, _ = normal_client.predict(test_loader)
        normal_accuracy = Metrics.accuracy(np.array(y_test), np.array(y_pred))

        print("Poison training")
        poisoned_model = SimpleLogisticRegression(num_features, 1)
        poisoned_client = MaliciousClient(poisoned_model, torch.nn.BCELoss(), torch.optim.Adam(poisoned_model.parameters(), lr=0.01), type=type_binary_classification, attack_type=attack_label_flip)
        poisoned_client.setDataLoader(train_loader, n_iters=1)
        poisoned_client.train()

        poisoned_y_pred, _ = poisoned_client.predict(test_loader)
        poisoned_accuracy = Metrics.accuracy(np.array(y_test), np.array(poisoned_y_pred))

        print("Normal accuracy: ", normal_accuracy)
        print("Poison accuracy: ", poisoned_accuracy)
        print("Diff accuracy: ", normal_accuracy - poisoned_accuracy)

    def train(self, train_loader, test_loader, attack_type, tag):
        print(f"{tag} Training")
        model = SimpleCNNClassifier()
        client = MaliciousClient(model, torch.nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=0.01), type=type_multi_classification, attack_type=attack_type)
        client.setArgs(flip_ratio=0.7, num_classes=10, poison_ratio=0.7, noise_level=10)
        client.setDataLoader(train_loader, n_iters=1)
        client.train()

        y_pred, _ = client.predict(test_loader)

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

        normal_accuracy = self.train(train_loader, test_loader, attack_type_none, "Normal")
        label_flip_accuracy = self.train(train_loader, test_loader, attack_label_flip, "Label Flip")
        sample_poisoned_accuracy = self.train(train_loader, test_loader, attack_sample_poison, "Sample Poison")
        ood_data_accuracy = self.train(train_loader, test_loader, attack_ood_data, "OOD Data")

        print("Normal accuracy: ", normal_accuracy)
        print("Label Flip accuracy: ", label_flip_accuracy)
        print("Sample Poison accuracy: ", sample_poisoned_accuracy)
        print("OOD Data accuracy: ", ood_data_accuracy)

class TestModelPoison(unittest.TestCase):
    def train(self, train_loader, test_loader, attack_type, tag):
        print(f"{tag} Training")
        model = SimpleCNNClassifier()
        client = MaliciousClient(model, torch.nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=0.01), type=type_multi_classification, attack_type=attack_type)
        client.setArgs(flip_ratio=0.7, num_classes=10, poison_ratio=0.7, noise_level=10)
        client.setDataLoader(train_loader, n_iters=1)
        client.train()

        y_pred, _ = client.predict(test_loader)

        y_test = []
        for _, y in test_loader:
            y_test += y.tolist()
        accuracy = Metrics.accuracy(np.array(y_test), np.array(y_pred))
        return accuracy
    
    def test_model_poison(self):
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

        normal_accuracy = self.train(train_loader, test_loader, attack_type_none, "Normal")
        gradient_poision_accuracy = self.train(train_loader, test_loader, attack_gradient_poison, "Gradient Poison")
        weight_poison_accuracy = self.train(train_loader, test_loader, attack_weight_poison, "Weight Poison")

        print("Normal accuracy: ", normal_accuracy)
        print("Gradient Poison accuracy: ", gradient_poision_accuracy)
        print("Weight Poison accuracy: ", weight_poison_accuracy)


class TestRobustAggr(unittest.TestCase):
    def _init_clients(self, n_clients, n_malicious_client, model_factory_json, attack_type):
        clients = []
        for _ in range(n_clients - n_malicious_client):
            model, model_type, optimizer, criterion = ModelFactory().create_model(model_factory_json)
            client = Client(model, criterion, optimizer, type=model_type)
            clients.append(client)
        for _ in range(n_malicious_client):
            model, model_type, optimizer, criterion = ModelFactory().create_model(model_factory_json)
            client = MaliciousClient(model, criterion, optimizer, type=model_type, attack_type=attack_type)
            clients.append(client)
        random.shuffle(clients)
        return clients
    
    def _init_server(self, model_factory_json):
        model, model_type, optimizer, criterion = ModelFactory().create_model(model_factory_json)
        return RobustAggrServer(model, optimizer, criterion, model_type)
    
    def _model_json(self):
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
                    "padding": 0,
                    "stride": 1
                },
                {
                    "type": "relu"
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
                    "stride": 1
                },
                {
                    "type": "relu"
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
    
    def _prepare(self, compare, attack_type):
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

        n_clients = 6
        n_malicious_client = 1 # assume poisoned client less than normal clients (1/3)
        n_rounds = 10
        n_iter = 1

        clients = self._init_clients(n_clients, n_malicious_client, self._model_json(), attack_type)
        model, model_type, optimizer, criterion = ModelFactory().create_model(self._model_json())
        if compare:
            server = Server(model, optimizer, criterion, model_type)
        else:
            server = RobustAggrServer(model, optimizer, criterion, model_type)
        
        for i in range(n_clients):
            clients[i].setDataLoader(train_loader, n_iter)
        server.setTestLoader(test_loader)

        return server, clients, n_rounds
    
    def normal_grad_compare(self):
        server, clients, n_rounds = self._prepare(True, attack_sample_poison)
        controller = FLController(server, clients)
        controller.train(n_rounds, mode_avg_grad)
        server.model_metric.summary()
        
    def test_median_grad_aggr(self):
        server, clients, n_rounds = self._prepare(False, attack_sample_poison)
        controller = MedianAggrFLController(server, clients)
        controller.train(n_rounds, mode_avg_grad)
        server.model_metric.summary()

    def test_trimmed_mean_grad_aggr(self):
        server, clients, n_rounds = self._prepare(False, attack_sample_poison)
        controller = TrimmedMeanAggrFLController(server, clients, trim_ratio=0.2)
        controller.train(n_rounds, mode_avg_grad)
        server.model_metric.summary()
    
    def normal_weight_compare(self):
        server, clients, n_rounds = self._prepare(True)
        controller = FLController(server, clients)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()

    def test_median_weight_aggr(self):
        server, clients, n_rounds = self._prepare(False)
        controller = MedianAggrFLController(server, clients)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()

    def test_trimmed_mean_weight_aggr(self):
        server, clients, n_rounds = self._prepare(False)
        controller = TrimmedMeanAggrFLController(server, clients, trim_ratio=0.2)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()