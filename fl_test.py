import torch
import torchvision
import torch.utils.data
import unittest
import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing
import random
import numpy as np
import os

from dl.simple_cnn_classifier import SimpleCNNClassifier
from fl.client import Client
from fl.server import Server
from fl.controller import FLController, mode_avg_grad, mode_avg_weight, mode_avg_vote
from fl.model_factory import ModelFactory
from fl.model_factory import type_regression, type_binary_classification, type_multi_classification
from fl.psi import SimplePSI
from fl.fed_prox.fed_prox_client import FedProxClient
from fl.fed_prox.fed_prox_controller import FedProxController
from fl.fed_nova.fed_nova_client import FedNovaClient
from fl.fed_nova.fed_nova_server import FedNovaServer
from fl.fed_nova.fed_nova_controller import FedNovaController
from fl.scaffold.scaffold_client import ScaffoldClient
from fl.scaffold.scaffold_controller import ScaffoldController
from fl.scaffold.scaffold_server import ScaffoldServer
from fl.maml.maml_client import MAMLClient
from fl.maml.maml_controller import MAMLController
from fl.per_fed_avg.per_fed_avg_controller import PerFedAvgController
from fl.per_fed_avg.per_fed_avg_client import PerFedAvgClient
from fl.fed_adapt_opt.fed_adapt_opt_controller import FedAdaptOptController
from fl.fed_adapt_opt.fed_adapt_opt_server import FedAdaptOptServer, type_fed_adagrad, type_fed_yogi, type_fed_adam
from fl.p_fed_me.p_fed_me_client import pFedMeClient
from fl.p_fed_me.p_fed_me_controller import pFedMeController
from fl.p_fed_me.p_fed_me_server import pFedMeServer
from fl.fed_em.fed_em_client import FedEMClient
from fl.fed_em.fed_em_controller import FedEMController


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

    def _filter_target_loader(self, loader, target_classes):
        new_dataset = []
        for data, label in loader.dataset:
            if label in target_classes:
                new_dataset.append((data, label))
        return torch.utils.data.DataLoader(new_dataset,
                                           batch_size=32,
                                           shuffle=False)

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

    def _init_dataloader(self, batch_size, regression=True):
        if regression:
            X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
            scaler = sklearn.preprocessing.StandardScaler()
            X = scaler.fit_transform(X)
            X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(
                y, dtype=torch.float32).view(-1, 1)
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                X, y, test_size=0.1, random_state=20)
        else:
            X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
            X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(
                y, dtype=torch.float32).view(-1, 1)
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                X, y, test_size=0.1, random_state=42)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test, y_test),
            batch_size=batch_size,
            shuffle=True)
        return train_loader, test_loader

    def _model_json(self, regression=True):
        if regression:
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
        else:
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
        return model_json

    def _init_clients(self, model_json, n_clients, train_loader, n_iter):
        clients = []
        for i in range(n_clients):
            model, model_type, optimizer, criterion = ModelFactory(
            ).create_model(model_json)
            client = Client(model, criterion, optimizer, type=model_type)
            client.setDataLoader(train_loader, n_iter)
            clients.append(client)
        return clients

    def _init_server(self, model_json, test_loader):
        model, model_type, optimizer, criterion = ModelFactory().create_model(
            model_json)
        server = Server(model, optimizer, criterion, type=model_type)
        server.setTestLoader(test_loader)
        return server

    def _prepare(self, regression=True, batch_size=100, n_clients=5, n_iter=1):

        train_loader, test_loader = self._init_dataloader(
            batch_size, regression)
        model_json = self._model_json(regression)
        clients = self._init_clients(model_json, n_clients, train_loader,
                                     n_iter)
        server = self._init_server(model_json, test_loader)

        return server, clients

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

        np.random.seed(0)
        torch.manual_seed(0)
        random.seed(0)


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

        n_clients = 2
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

        n_rounds = 1
        n_batch_size = 10
        n_iter = 100

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

    def test_mnist_fl_classification(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
        ])

        train_dataset = torchvision.datasets.MNIST(root='./data',
                                                   train=True,
                                                   download=True,
                                                   transform=transform)

        test_dataset = torchvision.datasets.MNIST(root='./data',
                                                  train=False,
                                                  download=True,
                                                  transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=32,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=32,
                                                  shuffle=False)

        model_json = {
            "model_type":
            type_multi_classification,
            "learning_rate":
            0.001,
            "optimizer":
            "adam",
            "criterion":
            "cross_entropy",
            "layers": [{
                "type": "conv2d",
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": 3,
                "padding": 1,
                "stride": 1
            }, {
                "type": "relu"
            }, {
                "type": "maxpool",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0
            }, {
                "type": "conv2d",
                "in_channels": 32,
                "out_channels": 64,
                "kernel_size": 3,
                "padding": 1,
                "stride": 1
            }, {
                "type": "relu"
            }, {
                "type": "maxpool",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0
            }, {
                "type": "reshape",
                "shape": [-1, 64 * 7 * 7]
            }, {
                "type": "linear",
                "in_features": 7 * 7 * 64,
                "out_features": 128
            },  {
                "type": "linear",
                "in_features": 128,
                "out_features": 10
            }, {
                "type": "softmax",
                "dim": 1
            }]
        }

        np.random.seed(0)
        torch.manual_seed(0)
        random.seed(0)

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

        n_rounds = 40
        n_iter = 1

        for client_id in range(n_clients):
            clients[client_id].setDataLoader(train_loader, n_iter)

        server.setTestLoader(test_loader)

        controller = FLController(server, clients)
        controller.train(n_rounds, mode_avg_weight)

        if not os.path.exists("./cache/model"):
            os.makedirs("./cache/model")
        torch.save(server.model.state_dict(), "./cache/model/mnist_model.pth")

        server.model_metric.summary()


    def test_cifar10_fl_classification(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
        ])

        train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                    train=True,
                                                    download=True,
                                                    transform=transform)

        test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                    train=False,
                                                    download=True,
                                                    transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=32,
                                                shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=32,
                                                shuffle=False)

        model_json = {
            "model_type": type_multi_classification,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "criterion": "cross_entropy",
            "layers": [{
                "type": "conv2d",
                "in_channels": 3,  # CIFAR-10 has 3 channels
                "out_channels": 32,
                "kernel_size": 3,
                "padding": 1,
                "stride": 1
            }, {
                "type": "relu"
            }, {
                "type": "maxpool",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0
            }, {
                "type": "conv2d",
                "in_channels": 32,
                "out_channels": 64,
                "kernel_size": 3,
                "padding": 1,
                "stride": 1
            }, {
                "type": "relu"
            }, {
                "type": "maxpool",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0
            }, {
                "type": "reshape",
                "shape": [-1, 64 * 8 * 8]  # Adjust shape to match the output size
            }, {
                "type": "linear",
                "in_features": 8 * 8 * 64,
                "out_features": 128
            }, {
                "type": "relu"
            }, {
                "type": "linear",
                "in_features": 128,
                "out_features": 10  # CIFAR-10 has 10 classes
            }]
        }

        np.random.seed(0)
        torch.manual_seed(0)
        random.seed(0)

        n_clients = 2
        clients = []
        for i in range(n_clients):
            # each client should have its own model
            model, model_type, optimizer, criterion = ModelFactory().create_model(model_json)
            client = Client(model, criterion, optimizer, type=model_type)
            clients.append(client)

        model, model_type, optimizer, criterion = ModelFactory().create_model(model_json)
        server = Server(model, optimizer, criterion, type=model_type)

        n_rounds = 50
        n_iter = 1

        for client_id in range(n_clients):
            clients[client_id].setDataLoader(train_loader, n_iter)

        server.setTestLoader(test_loader)

        controller = FLController(server, clients)
        controller.train(n_rounds, mode_avg_weight)

        if not os.path.exists("./cache/model"):
            os.makedirs("./cache/model")
        torch.save(server.model.state_dict(), "./cache/model/cifar10_model.pth")

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


class TestFedAdaptOpt(TestFL):

    def _init_server(self, model_json, test_loader, type_optim, lr, **kwargs):
        model, model_type, _, _ = ModelFactory().create_model(model_json)
        server = FedAdaptOptServer(model,
                                   type=model_type,
                                   type_optimizer=type_optim,
                                   lr=lr,
                                   **kwargs)
        server.setTestLoader(test_loader)
        return server

    def _prepare(self, regression, batch_size, n_clients, n_iter, type_optim,
                 lr, **kwargs):
        train_loader, test_loader = self._init_dataloader(
            batch_size, regression)
        model_json = self._model_json(regression)
        clients = self._init_clients(model_json, n_clients, train_loader,
                                     n_iter)
        server = self._init_server(model_json, test_loader, type_optim, lr,
                                   **kwargs)
        return server, clients

    def test_fed_adagrad_classification(self):
        server, clients = self._prepare(regression=False,
                                        batch_size=16,
                                        n_clients=5,
                                        n_iter=1,
                                        type_optim=type_fed_adagrad,
                                        lr=0.01,
                                        epsilon=1e-8)
        controller = FedAdaptOptController(server, clients)
        controller.train(n_rounds=10, mode=mode_avg_weight)
        server.model_metric.summary()

    def test_fed_yogi_classification(self):
        server, clients = self._prepare(regression=False,
                                        batch_size=16,
                                        n_clients=5,
                                        n_iter=1,
                                        type_optim=type_fed_yogi,
                                        lr=0.01,
                                        epsilon=1e-3,
                                        lamb=0.1,
                                        zeta=0.1)
        controller = FedAdaptOptController(server, clients)
        controller.train(n_rounds=10, mode=mode_avg_weight)
        server.model_metric.summary()

    def test_fed_adam_classification(self):
        server, clients = self._prepare(regression=False,
                                        batch_size=16,
                                        n_clients=5,
                                        n_iter=1,
                                        type_optim=type_fed_adam,
                                        lr=0.01,
                                        epsilon=1e-8,
                                        beta1=0.9,
                                        beta2=0.999)
        controller = FedAdaptOptController(server, clients)
        controller.train(n_rounds=10, mode=mode_avg_weight)
        server.model_metric.summary()


class TestFedProx(TestFL):

    def _init_clients(self, model_json, n_clients, train_loader, n_iter):
        clients = []
        for i in range(n_clients):
            model, model_type, optimizer, criterion = ModelFactory(
            ).create_model(model_json)
            client = FedProxClient(model,
                                   criterion,
                                   optimizer,
                                   type=model_type,
                                   mu=0.1)
            client.setDataLoader(train_loader, n_iter)
            clients.append(client)
        return clients

    def test_fed_prox_regression(self):
        server, clients = self._prepare(regression=True,
                                        batch_size=100,
                                        n_clients=5,
                                        n_iter=1)
        controller = FedProxController(server, clients)
        controller.train(n_rounds=10, mode=mode_avg_weight)
        server.model_metric.summary()

    def test_fed_prox_classification(self):
        server, clients = self._prepare(regression=False,
                                        batch_size=16,
                                        n_clients=5,
                                        n_iter=1)
        controller = FedProxController(server, clients)
        controller.train(n_rounds=2, mode=mode_avg_weight)
        server.model_metric.summary()


class TestFedNova(TestFL):

    def _init_clients(self, model_json, n_clients, train_loader, n_iter):
        clients = []
        for i in range(n_clients):
            model, model_type, optimizer, criterion = ModelFactory(
            ).create_model(model_json)
            client = FedNovaClient(model,
                                   criterion,
                                   optimizer,
                                   type=model_type)
            client.setDataLoader(train_loader, n_iter)
            clients.append(client)
        return clients

    def _init_server(self, model_json, test_loader):
        model, model_type, optimizer, criterion = ModelFactory().create_model(
            model_json)
        server = FedNovaServer(model, optimizer, criterion, type=model_type)
        server.setTestLoader(test_loader)
        return server

    def test_fed_nova_classification(self):
        server, clients = self._prepare(regression=False,
                                        batch_size=16,
                                        n_clients=5,
                                        n_iter=1)
        controller = FedNovaController(server, clients)
        controller.train(n_rounds=10, mode=mode_avg_weight)
        server.model_metric.summary()


class TestScalffold(TestFL):

    def _init_clients(self, model_json, n_clients, train_loader, n_iter):
        clients = []
        for i in range(n_clients):
            model, model_type, optimizer, criterion = ModelFactory(
            ).create_model(model_json)
            client = ScaffoldClient(model,
                                    criterion,
                                    optimizer,
                                    type=model_type,
                                    c_lr=0.01)
            client.setDataLoader(train_loader, n_iter)
            clients.append(client)
        return clients

    def _init_server(self, model_json, test_loader):
        model, model_type, optimizer, criterion = ModelFactory().create_model(
            model_json)
        server = ScaffoldServer(model, optimizer, criterion, type=model_type)
        server.setTestLoader(test_loader)
        return server

    def test_scaffold_classification(self):
        server, clients = self._prepare(regression=False,
                                        batch_size=16,
                                        n_clients=5,
                                        n_iter=1)
        controller = ScaffoldController(server, clients)
        controller.train(n_rounds=10, mode=mode_avg_weight)
        server.model_metric.summary()


class TestMAML(TestFL):

    def _init_task(self, loader):
        # assume we have two task, one for odd, one for even
        tasks = []
        odd_loader = self._filter_target_loader(loader, [1, 3, 5, 7, 9])
        # for simplicity, we use the same loader for both train and val
        tasks.append((odd_loader, odd_loader))
        even_loader = self._filter_target_loader(loader, [0, 2, 4, 6, 8])
        tasks.append((even_loader, even_loader))
        return tasks

    def _init_clients(self, model_json, n_clients, train_loader, n_iter):
        clients = []
        for i in range(n_clients):
            model, model_type, optimizer, criterion = ModelFactory(
            ).create_model(model_json)
            client = MAMLClient(model,
                                criterion,
                                optimizer,
                                inner_lr=0.01,
                                type=model_type)
            tasks = self._init_task(train_loader)
            client.setDataLoader(tasks, n_iter)
            clients.append(client)
        return clients

    def test_maml_classification(self):
        server, clients = self._prepare(regression=False,
                                        batch_size=16,
                                        n_clients=5,
                                        n_iter=10)
        controller = MAMLController(server, clients)
        controller.train(n_rounds=10, mode=mode_avg_weight)
        server.model_metric.summary()


class TestPerFedAvg(TestFL):

    def _init_clients(self, model_json, n_clients, train_loader, n_iter):
        clients = []
        for i in range(n_clients):
            model, model_type, optimizer, criterion = ModelFactory(
            ).create_model(model_json)
            client = PerFedAvgClient(model,
                                     criterion,
                                     optimizer,
                                     type=model_type)
            client.setDataLoader(train_loader, n_iter)
            clients.append(client)
        return clients

    def test_per_fed_avg_classification(self):
        server, clients = self._prepare(regression=False,
                                        batch_size=16,
                                        n_clients=5,
                                        n_iter=1)
        controller = PerFedAvgController(server, clients)
        controller.train(n_rounds=10, mode=mode_avg_weight)
        server.model_metric.summary()


class TestpFedMe(TestFL):

    def _init_clients(self, model_json, n_clients, train_loader, n_iter):
        clients = []
        for i in range(n_clients):
            model, model_type, optimizer, criterion = ModelFactory(
            ).create_model(model_json)
            client = pFedMeClient(model,
                                  criterion,
                                  optimizer,
                                  type=model_type,
                                  beta=0.1)
            client.setDataLoader(train_loader, n_iter)
            clients.append(client)
        return clients

    def _init_server(self, model_json, test_loader):
        model, model_type, optimizer, criterion = ModelFactory().create_model(
            model_json)
        server = pFedMeServer(model, optimizer, criterion, type=model_type)
        server.setTestLoader(test_loader)
        return server

    def test_p_fed_me_classification(self):
        server, clients = self._prepare(regression=False,
                                        batch_size=16,
                                        n_clients=5,
                                        n_iter=1)
        controller = pFedMeController(server, clients)
        controller.train(n_rounds=10, mode=mode_avg_weight)
        server.model_metric.summary()


class TestFedEM(TestFL):

    def _init_clients(self, model_json, n_clients, train_loader, n_iter):
        clients = []
        for _ in range(n_clients):
            n_models = 5  # assume each client has 5 models
            models = []
            _, model_type, _, criterion = ModelFactory().create_model(
                model_json)
            for _ in range(n_models):
                model, _, _, _ = ModelFactory().create_model(model_json)
                models.append(model)
            client = FedEMClient(models, criterion, type=model_type)
            client.setDataLoader(train_loader, n_iter)
            clients.append(client)
        return clients

    def test_fed_em_classification(self):
        server, clients = self._prepare(regression=False,
                                        batch_size=16,
                                        n_clients=5,
                                        n_iter=1)
        controller = FedEMController(server, clients)
        controller.train(n_rounds=5, mode=mode_avg_vote)
        server.model_metric.summary()


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
