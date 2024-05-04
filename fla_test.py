from fl.client import Client
from fl.server import Server
from fl.model_factory import ModelFactory
from dl.simple_cnn_classifier import SimpleCNNClassifier
import dl.metrics as Metrics

from fla.data_poison import DataPoison

import numpy as np
import torch
import torch.utils.data
import torchvision
import unittest

class TestDataPoison(unittest.TestCase):
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
