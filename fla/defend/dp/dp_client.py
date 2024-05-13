import torch
import torch.utils.data

import crypto.diff_privacy as DiffPrivacy
from crypto.diff_privacy import DPSGD
from fl.client import Client


class InputPerturbationClient(Client):

    def __init__(self, model, criterion, optimizer, type):
        super().__init__(model, criterion, optimizer, type)

    def setDataLoader(self,
                      train_loader: torch.utils.data.DataLoader,
                      n_iters,
                      epsilon=0.1,
                      delta=1e-5,
                      sensitivity=1.0):
        X, y = train_loader.dataset.tensors
        X = DiffPrivacy.gaussian_dp(X, epsilon, delta, sensitivity)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y),
            batch_size=train_loader.batch_size,
            shuffle=True)
        return super().setDataLoader(train_loader, n_iters)


class DPSGDClient(Client):

    def __init__(self, model, criterion, optimizer, type):
        super().__init__(model, criterion, optimizer, type)

    def train(self, round_idx=-1, sigma=0.1, clip_value=0.1, delta=1e-5):
        dp_sgd = DPSGD(self.model, sigma, clip_value, delta)
        dp_sgd.train(self.train_loader, self.n_iters)
