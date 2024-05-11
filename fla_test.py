from dl.simple_cnn_classifier import SimpleCNNClassifier
from dl.simple_logistic_regression import SimpleLogisticRegression
import dl.metrics as Metrics

from fl.client import Client
from fl.server import Server
from fl.controller import FLController, mode_avg_grad, mode_avg_weight, mode_avg_vote
from fl.model_factory import ModelFactory
from fl.model_factory import type_regression, type_binary_classification, type_multi_classification

from fla.malicious_client import MaliciousClient
from fla.malicious_client import attack_type_none, attack_sample_poison, attack_label_flip, attack_ood_data, attack_backdoor, attack_gradient_poison, attack_weight_poison
import fla.inference_attack as InferenceAttack
from fla.defend.robust_aggr.robust_aggr_server import RobustAggrServer
from fla.defend.robust_aggr.robust_aggr_controller import MedianAggrFLController, TrimmedMeanAggrFLController, KrumAggrFLController
from fla.defend.detection.anomaly_detection_server import AnomalyDetectionServer
from fla.defend.mpc.mpc_server import MPCServer
from fla.defend.mpc.mpc_client import MPCClient
from fla.defend.mpc.mpc_controller import MPCController

import numpy as np
import torch
import torch.utils.data
import torchvision
import sklearn.datasets
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
import random
import unittest
import matplotlib.pyplot as plt


class TestDataPoison(unittest.TestCase):

    def test_binary_label_flip(self):
        X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32).view(-1, 1)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.1, random_state=42)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=10,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test, y_test),
            batch_size=10,
            shuffle=False)

        num_features = X_train.shape[1]

        y_test = []
        for _, y in test_loader:
            y_test += y.tolist()

        print("Normal training")
        model = SimpleLogisticRegression(num_features, 1)
        normal_client = Client(model,
                               torch.nn.BCELoss(),
                               torch.optim.Adam(model.parameters(), lr=0.01),
                               type=type_binary_classification)
        normal_client.setDataLoader(train_loader, n_iters=1)
        normal_client.train()

        y_pred, _ = normal_client.predict(test_loader)
        normal_accuracy = Metrics.accuracy(np.array(y_test), np.array(y_pred))

        print("Poison training")
        poisoned_model = SimpleLogisticRegression(num_features, 1)
        poisoned_client = MaliciousClient(poisoned_model,
                                          torch.nn.BCELoss(),
                                          torch.optim.Adam(
                                              poisoned_model.parameters(),
                                              lr=0.01),
                                          type=type_binary_classification,
                                          attack_type=attack_label_flip)
        poisoned_client.setDataLoader(train_loader, n_iters=1)
        poisoned_client.train()

        poisoned_y_pred, _ = poisoned_client.predict(test_loader)
        poisoned_accuracy = Metrics.accuracy(np.array(y_test),
                                             np.array(poisoned_y_pred))

        print("Normal accuracy: ", normal_accuracy)
        print("Poison accuracy: ", poisoned_accuracy)
        print("Diff accuracy: ", normal_accuracy - poisoned_accuracy)

    def train(self, train_loader, test_loader, attack_type, tag):
        print(f"{tag} Training")
        model = SimpleCNNClassifier()
        client = MaliciousClient(model,
                                 torch.nn.CrossEntropyLoss(),
                                 torch.optim.Adam(model.parameters(), lr=0.01),
                                 type=type_multi_classification,
                                 attack_type=attack_type)
        client.setArgs(flip_ratio=0.7,
                       num_classes=10,
                       poison_ratio=0.7,
                       noise_level=10)
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
            torchvision.transforms.Normalize((0.5, ), (0.5, ))
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
                                                   batch_size=64,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=64,
                                                  shuffle=False)

        normal_accuracy = self.train(train_loader, test_loader,
                                     attack_type_none, "Normal")
        label_flip_accuracy = self.train(train_loader, test_loader,
                                         attack_label_flip, "Label Flip")
        sample_poisoned_accuracy = self.train(train_loader, test_loader,
                                              attack_sample_poison,
                                              "Sample Poison")
        ood_data_accuracy = self.train(train_loader, test_loader,
                                       attack_ood_data, "OOD Data")

        print("Normal accuracy: ", normal_accuracy)
        print("Label Flip accuracy: ", label_flip_accuracy)
        print("Sample Poison accuracy: ", sample_poisoned_accuracy)
        print("OOD Data accuracy: ", ood_data_accuracy)


class TestModelPoison(unittest.TestCase):

    def train(self, train_loader, test_loader, attack_type, tag):
        print(f"{tag} Training")
        model = SimpleCNNClassifier()
        client = MaliciousClient(model,
                                 torch.nn.CrossEntropyLoss(),
                                 torch.optim.Adam(model.parameters(), lr=0.01),
                                 type=type_multi_classification,
                                 attack_type=attack_type)
        client.setArgs(flip_ratio=0.7,
                       num_classes=10,
                       poison_ratio=0.7,
                       noise_level=10)
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
            torchvision.transforms.Normalize((0.5, ), (0.5, ))
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
                                                   batch_size=64,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=64,
                                                  shuffle=False)

        normal_accuracy = self.train(train_loader, test_loader,
                                     attack_type_none, "Normal")
        gradient_poision_accuracy = self.train(train_loader, test_loader,
                                               attack_gradient_poison,
                                               "Gradient Poison")
        weight_poison_accuracy = self.train(train_loader, test_loader,
                                            attack_weight_poison,
                                            "Weight Poison")

        print("Normal accuracy: ", normal_accuracy)
        print("Gradient Poison accuracy: ", gradient_poision_accuracy)
        print("Weight Poison accuracy: ", weight_poison_accuracy)


class TestFLA(unittest.TestCase):

    def _init_clients(self, n_clients, n_malicious_client, model_factory_json,
                      attack_type):
        clients = []
        for _ in range(n_clients - n_malicious_client):
            model, model_type, optimizer, criterion = ModelFactory(
            ).create_model(model_factory_json)
            client = Client(model, criterion, optimizer, type=model_type)
            clients.append(client)
        for _ in range(n_malicious_client):
            model, model_type, optimizer, criterion = ModelFactory(
            ).create_model(model_factory_json)
            client = MaliciousClient(model,
                                     criterion,
                                     optimizer,
                                     type=model_type,
                                     attack_type=attack_type)
            client.setArgs(flip_ratio=0.7,
                           num_classes=10,
                           poison_ratio=0.7,
                           noise_level=10)
            clients.append(client)
        random.shuffle(clients)
        return clients

    def _init_server(self, model_factory_json):
        model, model_type, optimizer, criterion = ModelFactory().create_model(
            model_factory_json)
        return Server(model, optimizer, criterion, model_type)

    def _model_json(self):
        return {
            "model_type":
            type_multi_classification,
            "learning_rate":
            0.01,
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
            }, {
                "type": "relu"
            }, {
                "type": "linear",
                "in_features": 128,
                "out_features": 10
            }, {
                "type": "softmax",
                "dim": 1
            }]
        }

    def _init_dataloader(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, ), (0.5, ))
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
        return train_loader, test_loader

    def _prepare(self, attack_type):
        train_loader, test_loader = self._init_dataloader()

        n_clients = 10
        n_malicious_client = 2  # assume poisoned client less than normal clients (1/3)
        n_rounds = 10
        n_iter = 1

        clients = self._init_clients(n_clients, n_malicious_client,
                                     self._model_json(), attack_type)
        server = self._init_server(self._model_json())

        for i in range(n_clients):
            clients[i].setDataLoader(train_loader, n_iter)
        server.setTestLoader(test_loader)

        return server, clients, n_rounds


class TestRobustAggr(TestFLA):

    def _init_server(self, model_factory_json):
        model, model_type, optimizer, criterion = ModelFactory().create_model(
            model_factory_json)
        return RobustAggrServer(model, optimizer, criterion, model_type)

    def normal_grad_compare(self):
        server, clients, n_rounds = self._prepare(attack_sample_poison)
        controller = FLController(server, clients)
        controller.train(n_rounds, mode_avg_grad)
        server.model_metric.summary()

    def test_median_grad_aggr(self):
        server, clients, n_rounds = self._prepare(attack_sample_poison)
        controller = MedianAggrFLController(server, clients)
        controller.train(n_rounds, mode_avg_grad)
        server.model_metric.summary()

    def test_trimmed_mean_grad_aggr(self):
        server, clients, n_rounds = self._prepare(attack_sample_poison)
        controller = TrimmedMeanAggrFLController(server,
                                                 clients,
                                                 trim_ratio=0.2)
        controller.train(n_rounds, mode_avg_grad)
        server.model_metric.summary()

    def normal_weight_compare(self):
        server, clients, n_rounds = self._prepare(attack_weight_poison)
        controller = FLController(server, clients)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()

    def test_median_weight_aggr(self):
        server, clients, n_rounds = self._prepare(attack_weight_poison)
        controller = MedianAggrFLController(server, clients)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()

    def test_trimmed_mean_weight_aggr(self):
        server, clients, n_rounds = self._prepare(attack_weight_poison)
        controller = TrimmedMeanAggrFLController(server,
                                                 clients,
                                                 trim_ratio=0.2)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()

    def normal_backdoor_attack(self):
        server, clients, n_rounds = self._prepare(attack_backdoor)
        controller = FLController(server, clients)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()

        print(
            "The odd round will trigger the backdoor attack, the accuracy will be lower than even round."
        )

    def test_krum_weight_aggr(self):
        server, clients, n_rounds = self._prepare(attack_backdoor)
        controller = KrumAggrFLController(
            server, clients,
            n_malicious=2)  # asume admin think there is 2 malicious client
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()


class TestAnomalyDetection(TestFLA):

    def _init_server(self, model_factory_json):
        model, model_type, optimizer, criterion = ModelFactory().create_model(
            model_factory_json)
        return AnomalyDetectionServer(model,
                                      optimizer,
                                      criterion,
                                      model_type,
                                      k=2,
                                      min_samples=2)

    def test_gradient_anomaly_detection(self):
        server, clients, n_rounds = self._prepare(attack_gradient_poison)
        controller = FLController(server, clients)
        controller.train(n_rounds, mode_avg_grad)
        server.model_metric.summary()

    def test_weight_anomaly_detection(self):
        server, clients, n_rounds = self._prepare(attack_weight_poison)
        controller = FLController(server, clients)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()


class TestMPC(TestFLA):

    def _init_clients(self, n_clients, n_malicious_client, model_factory_json,
                      attack_type):
        clients = []
        for _ in range(n_clients):
            model, model_type, optimizer, criterion = ModelFactory(
            ).create_model(model_factory_json)
            client = MPCClient(model, criterion, optimizer, type=model_type)
            clients.append(client)
        random.shuffle(clients)
        return clients

    def _init_server(self, model_factory_json):
        model, model_type, optimizer, criterion = ModelFactory().create_model(
            model_factory_json)
        return MPCServer(model, optimizer, criterion, model_type, 5)

    def test_mpc_grad_aggr(self):
        server, clients, n_rounds = self._prepare(attack_type_none)
        controller = MPCController(server, clients)
        controller.train(n_rounds, mode_avg_grad)
        server.model_metric.summary()

    def test_mpc_weight_aggr(self):
        server, clients, n_rounds = self._prepare(attack_type_none)
        controller = MPCController(server, clients)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()


class TestModelInversion(TestFLA):

    def display_inverted_images(self, inverted_images, n_images):
        plt.figure(figsize=(20, 4))
        for i in range(n_images):
            ax = plt.subplot(2, n_images, i + 1)
            img = inverted_images[i].moveaxis(0, -1)
            img = (img - img.min()) / (img.max() - img.min())
            plt.imshow(img)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def test_model_inversion(self):
        server, clients, n_rounds = self._prepare(attack_type_none)
        controller = FLController(server, clients)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()

        target_class = 0
        n_samples = 10
        input_shape = (1, 28, 28)
        n_step = 100
        lr = 0.01
        synthetic_inputs = InferenceAttack.model_inversion_attack(
            controller, target_class, n_samples, input_shape, n_step, lr)
        self.display_inverted_images(synthetic_inputs, n_samples)


class TestLabelInference(TestFLA):

    def _exclude_target_dataset(self, dataset, target_class):
        new_dataset = []
        for data, label in dataset:
            if label != target_class:
                new_dataset.append((data, label))
        return new_dataset

    def _filter_target_loader(self, loader, target_class):
        new_dataset = []
        for data, label in loader.dataset:
            if label == target_class:
                new_dataset.append((data, label))
        return torch.utils.data.DataLoader(new_dataset,
                                           batch_size=32,
                                           shuffle=False)

    def _init_dataloader(self, exclude_target_class):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, ), (0.5, ))
        ])

        train_dataset = torchvision.datasets.MNIST(root='./data',
                                                   train=True,
                                                   download=True,
                                                   transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data',
                                                  train=False,
                                                  download=True,
                                                  transform=transform)

        train_dataset = self._exclude_target_dataset(train_dataset,
                                                     exclude_target_class)
        test_dataset = self._exclude_target_dataset(test_dataset,
                                                    exclude_target_class)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=32,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=32,
                                                  shuffle=False)

        return train_loader, test_loader

    def _prepare(self, attack_type, exclude_target_class):
        train_loader, test_loader = self._init_dataloader(exclude_target_class)

        n_clients = 5
        n_malicious_client = 0
        n_rounds = 1
        n_iter = 1

        clients = self._init_clients(n_clients, n_malicious_client,
                                     self._model_json(), attack_type)
        server = self._init_server(self._model_json())

        for i in range(n_clients):
            clients[i].setDataLoader(train_loader, n_iter)
        server.setTestLoader(test_loader)

        return server, clients, n_rounds

    def train_model(self, target_class):
        server, clients, n_rounds = self._prepare(attack_type_none,
                                                  target_class)
        controller = FLController(server, clients)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()
        return controller

    def test_label_inference(self):
        target_class = 0  # assume we want to detect target class 0 whether exists
        threshold = 0.1

        compared_model = self.train_model(-1)
        attacked_model = self.train_model(target_class)

        attack_loader = self._filter_target_loader(
            compared_model.server.test_loader, target_class)

        compared_result = InferenceAttack.label_inference_attack(
            compared_model, target_class, attack_loader, threshold)
        attacked_result = InferenceAttack.label_inference_attack(
            attacked_model, target_class, attack_loader, threshold)
        print(f"Compared model result: Target {target_class} exists -",
              compared_result)
        print(f"Attacked model result: Target {target_class} exists -",
              attacked_result)


class TestFeatureInference(TestFLA):

    def _init_dataloader(self):
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32).view(-1, 1)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.1, random_state=10)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=100,
            shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test, y_test),
            batch_size=100,
            shuffle=False)
        return train_loader, test_loader

    def _model_json(self):
        return {
            "model_type": type_regression,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "criterion": "mse",
            "layers": [{
                "type": "linear",
                "in_features": 8,
                "out_features": 1
            }]
        }

    def _prepare(self):
        train_loader, test_loader = self._init_dataloader()

        n_clients = 5
        n_malicious_client = 0
        n_rounds = 1
        n_iter = 1

        clients = self._init_clients(n_clients, n_malicious_client,
                                     self._model_json(), attack_type_none)
        server = self._init_server(self._model_json())

        for i in range(n_clients):
            clients[i].setDataLoader(train_loader, n_iter)
        server.setTestLoader(test_loader)

        return server, clients, n_rounds

    def test_feature_inference(self):
        server, clients, n_rounds = self._prepare()
        controller = FLController(server, clients)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()

        input_shape = (8, )
        n_samples = 10
        feature_index = 0  # assume we want to detect feature 0 whether exists
        delta = 0.1
        threshold = 0.05

        result = InferenceAttack.feature_inference_attack(
            controller, input_shape, n_samples, feature_index, delta,
            threshold)
        print(f"Feature Inference Attack: feature {feature_index} exists -",
              result)
