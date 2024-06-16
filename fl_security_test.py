import torch.utils.data.dataloader
from dl.simple_cnn_classifier import SimpleCNNClassifier
from dl.simple_logistic_regression import SimpleLogisticRegression
import dl.metrics as Metrics

from fl.client import Client
from fl.server import Server
from fl.controller import FLController, mode_avg_grad, mode_avg_weight, mode_avg_vote
from fl.model_factory import ModelFactory
from fl.model_factory import type_regression, type_binary_classification, type_multi_classification

from fl_security.attack.training.malicious_controller import MaliciousFLController
from fl_security.attack.training.malicious_client import MaliciousClient
from fl_security.attack.training.malicious_client import attack_type_none, attack_sample_poison, attack_label_flip, attack_ood_data, attack_backdoor, attack_gradient_poison, attack_weight_poison
from fl_security.attack.training.dlg import DLG
import fl_security.attack.inference.common_inference as InferenceAttack
from fl_security.attack.inference.membership_inference import MembershipInferenceAttack
from fl_security.defend.robust_aggr.robust_aggr_server import RobustAggrServer
from fl_security.defend.robust_aggr.robust_aggr_controller import MedianAggrFLController, TrimmedMeanAggrFLController, KrumAggrFLController
from fl_security.defend.detection.anomaly_detection_server import AnomalyDetectionServer
from fl_security.defend.mpc.mpc_server import MPCServer
from fl_security.defend.mpc.mpc_client import MPCClient
from fl_security.defend.mpc.mpc_controller import MPCController
from fl_security.defend.dp.dp_server import OutputPerturbationServer
from fl_security.defend.dp.dp_client import InputPerturbationClient, DPSGDClient
from fl_security.defend.dp.dp_controller import OutputPerturbationFLController, InputPerturbationFLController, DPSGDFLController
from fl_security.defend.feature_squeeze.feature_squeeze_client import FeatureSqueezeClient

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
from tqdm import tqdm


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


class TestMembershipInferenceAttack(unittest.TestCase):

    def test_membership_inference_attack(self):
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

        model = SimpleCNNClassifier()
        model.fit(train_loader, learning_rate=0.01, n_iters=1)
        y_pred, y_prob = model.predict(test_loader)
        y_test = []
        for _, y in test_loader:
            y_test += y.tolist()
        accuracy = Metrics.accuracy(np.array(y_test), np.array(y_pred))
        print("Target model accuracy: ", accuracy)

        attacker = MembershipInferenceAttack(model,
                                             input_shape=(1, 28, 28),
                                             n_classes=10)
        membership_dataset, non_membership_dataset = attacker.create_shadow_dataset(
            1000)
        member_loader = torch.utils.data.DataLoader(membership_dataset,
                                                    batch_size=100,
                                                    shuffle=True)
        non_member_loader = torch.utils.data.DataLoader(non_membership_dataset,
                                                        batch_size=100,
                                                        shuffle=True)
        attacker.train_shadow_model(member_loader, non_member_loader, 10)
        attack_accuracy = attacker.attack(member_loader, non_member_loader, 10,
                                          0.8)
        print("Membership Inference Attack Accuracy: ", attack_accuracy)


class TestClassificationFLA(unittest.TestCase):

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

    def _init_dataloader(self,
                         exclude_target_class=-1,
                         filter_target_class=-1):
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

        if exclude_target_class != -1:
            train_dataset = self._exclude_target_dataset(
                train_dataset, exclude_target_class)
            test_dataset = self._exclude_target_dataset(
                test_dataset, exclude_target_class)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=32,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=32,
                                                  shuffle=False)

        if filter_target_class != -1:
            train_loader = self._filter_target_loader(train_loader,
                                                      filter_target_class)
            test_loader = self._filter_target_loader(test_loader,
                                                     filter_target_class)

        return train_loader, test_loader

    def _prepare(self, attack_type, exclude_target_class=-1):
        train_loader, test_loader = self._init_dataloader(exclude_target_class)

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

    def train_no_attack_model(self, exclude_target_class=-1):
        server, clients, n_rounds = self._prepare(attack_type_none,
                                                  exclude_target_class)
        controller = FLController(server, clients)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()
        return controller


class TestRegressionFLA(TestClassificationFLA):

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


class TestRobustAggr(TestClassificationFLA):

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


class TestAnomalyDetection(TestClassificationFLA):

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


class TestMPC(TestClassificationFLA):

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


class TestModelInversion(TestClassificationFLA):

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


class TestLabelInference(TestClassificationFLA):

    def test_label_inference(self):
        target_class = 0  # assume we want to detect target class 0 whether exists
        threshold = 0.1

        compared_model = self.train_no_attack_model(-1)
        attacked_model = self.train_no_attack_model(target_class)

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


class TestFeatureInference(TestRegressionFLA):

    def test_feature_inference(self):
        server, clients, n_rounds = self._prepare()
        controller = FLController(server, clients)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()

        input_shape = (8, )
        n_samples = 100
        feature_index = 0  # assume we want to detect feature 0 whether exists
        delta = 1.0
        threshold = 0.05

        result = InferenceAttack.feature_inference_attack(
            controller, input_shape, n_samples, feature_index, delta,
            threshold)
        print(f"Feature Inference Attack: feature {feature_index} exists -",
              result)


class TestFeatureSqueeze(TestRegressionFLA):

    def _init_clients(self, n_clients, n_malicious_client, model_factory_json,
                      attack_type):
        clients = []
        for _ in range(n_clients):
            model, model_type, optimizer, criterion = ModelFactory(
            ).create_model(model_factory_json)
            client = FeatureSqueezeClient(model,
                                          criterion,
                                          optimizer,
                                          type=model_type)
            client.set_bit_depth(3)
            clients.append(client)
        return clients

    def test_feature_squeeze(self):
        server, clients, n_rounds = self._prepare()
        controller = FLController(server, clients)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()

        input_shape = (8, )
        n_samples = 100
        feature_index = 0  # assume we want to detect feature 0 whether exists
        delta = 1.0
        threshold = 0.05

        result = InferenceAttack.feature_inference_attack(
            controller, input_shape, n_samples, feature_index, delta,
            threshold)
        print(f"Feature Inference Attack: feature {feature_index} exists -",
              result)


class TestSampleInference(TestRegressionFLA):

    def _infer_sample(self, controller, sample):
        input_shape = (8, )
        n_samples = 10
        threshold = 2
        return InferenceAttack.sample_inference_attack(controller, sample,
                                                       input_shape, n_samples,
                                                       threshold)

    def test_sample_inference(self):
        server, clients, n_rounds = self._prepare()
        controller = FLController(server, clients)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()

        random_sample = (torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                       0.8]), torch.tensor([0.9]))
        target_sample = clients[0].train_loader.dataset[0]

        random_result = self._infer_sample(controller, random_sample)
        target_result = self._infer_sample(controller, target_sample)

        print("Random Inference Attack: This sample exists -", random_result)
        print("Target Inference Attack: This sample exists -", target_result)


class TestDP(TestSampleInference):

    type_output_perturbation = "output_perturbation"
    type_input_perturbation = "input_perturbation"
    type_sgd_perturbation = "sgd_perturbation"

    def _init_server(self, model_factory_json, type):
        model, model_type, optimizer, criterion = ModelFactory().create_model(
            model_factory_json)
        if type == TestDP.type_output_perturbation:
            return OutputPerturbationServer(model, optimizer, criterion,
                                            model_type)
        else:
            return Server(model, optimizer, criterion, model_type)

    def _init_clients(self, n_clients, n_malicious_client, model_factory_json,
                      attack_type, type):
        clients = []
        for _ in range(n_clients):
            model, model_type, optimizer, criterion = ModelFactory(
            ).create_model(model_factory_json)
            if type == TestDP.type_input_perturbation:
                client = InputPerturbationClient(model,
                                                 criterion,
                                                 optimizer,
                                                 type=model_type)
            elif type == TestDP.type_sgd_perturbation:
                client = DPSGDClient(model,
                                     criterion,
                                     optimizer,
                                     type=model_type)
            else:
                client = Client(model, criterion, optimizer, type=model_type)
            clients.append(client)
        return clients

    def _prepare(self, type):
        train_loader, test_loader = self._init_dataloader()

        n_clients = 10
        n_malicious_client = 2
        n_rounds = 1
        n_iter = 1

        clients = self._init_clients(n_clients,
                                     n_malicious_client,
                                     self._model_json(),
                                     attack_type_none,
                                     type=type)
        server = self._init_server(self._model_json(), type=type)

        for i in range(n_clients):
            if type == TestDP.type_input_perturbation:
                clients[i].setDataLoader(train_loader,
                                         n_iter,
                                         epsilon=1,
                                         delta=1,
                                         sensitivity=0.001)
            else:
                clients[i].setDataLoader(train_loader, n_iter)
        server.setTestLoader(test_loader)

        return server, clients, n_rounds

    def test_output_perturbation(self):
        server, clients, n_rounds = self._prepare(
            type=TestDP.type_output_perturbation)
        controller = OutputPerturbationFLController(server,
                                                    clients,
                                                    epsilon=0.1)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()

        target_sample = clients[0].train_loader.dataset[0]
        target_result = self._infer_sample(controller, target_sample)
        print("DP Defend Target Inference Attack Result:",
              target_result == False)

    def test_input_perturbation(self):
        server, clients, n_rounds = self._prepare(
            type=TestDP.type_input_perturbation)
        controller = InputPerturbationFLController(server, clients)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()

        target_sample = clients[0].train_loader.dataset[0]
        target_result = self._infer_sample(controller, target_sample)
        print("DP Defend Target Inference Attack Result:",
              target_result == False)

    def test_sgd_perturbation(self):
        server, clients, n_rounds = self._prepare(
            type=TestDP.type_sgd_perturbation)
        controller = DPSGDFLController(server,
                                       clients,
                                       sigma=0.1,
                                       clip_value=0.1,
                                       delta=1e-5)
        controller.train(n_rounds, mode_avg_weight)
        server.model_metric.summary()

        target_sample = clients[0].train_loader.dataset[0]
        target_result = self._infer_sample(controller, target_sample)
        print("DP Defend Target Inference Attack Result:",
              target_result == False)


class TestLeakageAttack(unittest.TestCase):   

    def test_dlg(self):

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        datasets = torchvision.datasets.CIFAR100(root="./data", download=True, train=False, transform=transform)

        test_image = datasets[0][0]
        test_label = torch.tensor([datasets[0][1]], dtype=torch.long)

        onehot_test_label = DLG.Utils.label_to_onehot(test_label, num_classes=100)

        np.random.seed(1234)
        torch.manual_seed(1234)

        target_model = DLG.LeNet()
        leaked_grads = target_model.leak_grads(test_image.unsqueeze(0), onehot_test_label.unsqueeze(0))

        attacker = DLG(target_model, (3, 32, 32), 100)
        reconstructed_data = attacker.reconstruct_inputs_from_grads(leaked_grads, n_iter=1000)

        attacker.visualize(test_image, reconstructed_data[0])
