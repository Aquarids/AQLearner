import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm

from fl.client import Client


class FeatureSqueezeClient(Client):

    def __init__(self, model: torch.nn.Module, criterion, optimizer, type):
        super(FeatureSqueezeClient, self).__init__(model, criterion, optimizer,
                                                   type)
        self.bit_depth = 3

    def set_bit_depth(self, bit_depth):
        self.bit_depth = bit_depth

    def setDataLoader(self, train_loader, n_iters=10):
        self.n_iters = n_iters
        self.train_loader = self.squeeze_features(train_loader, self.bit_depth)

    def squeeze_features(self, loader, bit_depth):
        squeezed_samples = []
        squeezed_targets = []

        progress_bar = tqdm(range(len(loader)), desc='Squeezing features')
        for data, target in loader:
            squeezed_data = self.bit_depth_reduction(data, bit_depth)
            squeezed_samples.append(squeezed_data)
            squeezed_targets.append(target)
            progress_bar.update(1)
        progress_bar.close()

        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.cat(squeezed_samples, dim=0),
            torch.cat(squeezed_targets, dim=0)),
                                           batch_size=loader.batch_size,
                                           shuffle=False)

    def bit_depth_reduction(self, input, bit_depth):
        max_val = 2**bit_depth - 1
        input_min = input.min(dim=0, keepdim=True).values
        input_max = input.max(dim=0, keepdim=True).values
        input_norm = (input - input_min) / (input_max - input_min)
        input_reduced = np.round(input_norm * max_val) / max_val
        input_squeezed = input_reduced * (input_max - input_min) + input_min
        return input_squeezed
