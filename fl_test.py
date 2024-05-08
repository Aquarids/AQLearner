import torch
import torch.utils.data
import unittest
import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing

from fl.client import Client
from fl.server import Server
from fl.controller import FLController, mode_avg_grad, mode_avg_weight, mode_avg_vote
from fl.model_factory import ModelFactory
from fl.model_factory import type_regression, type_binary_classification, type_multi_classification
from fl.psi import SimplePSI


class TestModelFactory(unittest.TestCase):

    def test_model_factory(self):
        model_factory = ModelFactory()
        model_params = {
            "model_type":
            "regression",
            "learning_rate":
            0.001,
            "optimizer":
            "sgd",
            "criterion":
            "mse",
            "layers": [{
                "type": "linear",
                "in_features": 10,
                "out_features": 5
            }, {
                "type": "linear",
                "in_features": 5,
                "out_features": 1
            }]
        }
        model, model_type, optimizer, criterion = model_factory.create_model(
            model_params)

        print("Model Type: ", model_type)
        print("Model Detail: ", model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params}")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(
                    f"Layer: {name}, Size: {param.size()}, Values: {param.data}"
                )
        print("Optimizer: ", optimizer)
        print("criterion: ", criterion)


class TestFL(unittest.TestCase):

    def splite_data(self, X, y, n_clients):
        # the data should owned by the clinets themselves rather than the server, here just for showing the concept
        total_samples = len(X)
        samples_per_client = total_samples // n_clients
        total_samples_used = samples_per_client * n_clients

        X_adjusted = X[:total_samples_used]
        y_adjusted = y[:total_samples_used]

        X_clients = torch.split(X_adjusted, samples_per_client)
        y_clients = torch.split(y_adjusted, samples_per_client)

        train_loader_clients = []
        for client_id in range(n_clients):
            train_loader_clients.append(
                torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                    X_clients[client_id], y_clients[client_id]),
                                            batch_size=1,
                                            shuffle=True))
        return train_loader_clients

    def test_fl_regression(self, mode):
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
        scaler = sklearn.preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32).view(-1, 1)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.1, random_state=20)

        model_json = {
            "model_type":
            type_regression,
            "learning_rate":
            0.01,
            "optimizer":
            "adam",
            "criterion":
            "mse",
            "layers": [{
                "type": "linear",
                "in_features": 8,
                "out_features": 128
            }, {
                "type": "relu"
            }, {
                "type": "linear",
                "in_features": 128,
                "out_features": 64
            }, {
                "type": "relu"
            }, {
                "type": "linear",
                "in_features": 64,
                "out_features": 32
            }, {
                "type": "relu"
            }, {
                "type": "linear",
                "in_features": 32,
                "out_features": 1
            }]
        }

        n_clients = 5
        clients = []
        for i in range(n_clients):
            # each client should have its own model
            model, model_type, optimizer, criterion = ModelFactory(
            ).create_model(model_json)
            client = Client(model, criterion, optimizer, type=model_type)
            clients.append(client)

        model, model_type, optimizer, criterion = ModelFactory().create_model(
            model_json)
        server = Server(model, optimizer, criterion, type=model_type)

        n_rounds = 10
        n_batch_size = 100
        n_iter = 1

        train_loader_clients = self.splite_data(X_train, y_train, n_clients)
        for client_id in range(n_clients):
            clients[client_id].setDataLoader(train_loader_clients[client_id],
                                             n_iter)

        server.setTestLoader(
            torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                X_test, y_test),
                                        batch_size=n_batch_size,
                                        shuffle=True))

        controller = FLController(server, clients)
        controller.train(n_rounds, mode)

        server.model_metric.summary()

    def test_fl_classification(self, mode):
        X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32).view(-1, 1)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.1, random_state=42)

        model_json = {
            "model_type":
            type_binary_classification,
            "learning_rate":
            0.01,
            "optimizer":
            "adam",
            "criterion":
            "bce",
            "layers": [{
                "type": "linear",
                "in_features": 30,
                "out_features": 16
            }, {
                "type": "relu"
            }, {
                "type": "linear",
                "in_features": 16,
                "out_features": 8
            }, {
                "type": "relu"
            }, {
                "type": "linear",
                "in_features": 8,
                "out_features": 1
            }, {
                "type": "sigmoid"
            }]
        }
        model, model_type, optimizer, criterion = ModelFactory().create_model(
            model_json)

        n_clients = 10
        clients = []
        for i in range(n_clients):
            # each client should have its own model
            model, model_type, optimizer, criterion = ModelFactory(
            ).create_model(model_json)
            client = Client(model, criterion, optimizer, type=model_type)
            clients.append(client)

        model, model_type, optimizer, criterion = ModelFactory().create_model(
            model_json)
        server = Server(model, optimizer, criterion, type=model_type)

        n_rounds = 10
        n_batch_size = 16
        n_iter = 10

        train_loader_clients = self.splite_data(X_train, y_train, n_clients)
        for client_id in range(n_clients):
            clients[client_id].setDataLoader(train_loader_clients[client_id],
                                             n_iter)

        server.setTestLoader(
            torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                X_test, y_test),
                                        batch_size=n_batch_size,
                                        shuffle=True))

        controller = FLController(server, clients)
        controller.train(n_rounds, mode)

        server.model_metric.summary()

    def test_avg_grad_regression(self):
        self.test_fl_regression(mode_avg_grad)

    def test_avg_grad_classification(self):
        self.test_fl_classification(mode_avg_grad)

    def test_avg_weights_regression(self):
        self.test_fl_regression(mode_avg_weight)

    def test_avg_weights_classification(self):
        self.test_fl_classification(mode_avg_weight)

    def test_avg_votes_regression(self):
        self.test_fl_regression(mode_avg_vote)

    def test_avg_votes_classification(self):
        self.test_fl_classification(mode_avg_vote)


class TestPSI(unittest.TestCase):

    def test_psi(self):
        client_features = {
            "client_1": {"age", "height", "weight"},
            "client_2": {"height", "weight", "blood_type"},
        }

        # assume client_1 ask client_2 for the intersection of features
        psi = SimplePSI()
        psi.build_dic(client_features["client_1"])
        # client_2 should hash the features before sending to client_1
        client_2_hashed_features = [
            hash(feature) for feature in client_features["client_2"]
        ]
        common_features = psi.psi(client_2_hashed_features)
        print("Common Features: ", common_features)


if __name__ == '__main__':
    unittest.main()
