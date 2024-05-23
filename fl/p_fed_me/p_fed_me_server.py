from torch.nn.modules import Module
from fl.server import Server

import torch


class pFedMeServer(Server):

    def __init__(self, model: Module, optimizer, criterion, type):
        super().__init__(model, optimizer, criterion, type)

    def calculate_weights(self, weights):
        return self.weight_median_aggr(weights)

    def weight_median_aggr(self, weights):
        if weights is None or len(weights) == 0:
            return None

        new_weights = {}

        if len(weights) > 0 and isinstance(weights[0], dict):
            keys = weights[0].keys()
        else:
            return None

        for key in keys:
            param_group = [
                client_weights[key] for client_weights in weights
                if key in client_weights
            ]
            stacked_weights = torch.stack(param_group)
            median = torch.median(stacked_weights, dim=0).values
            new_weights[key] = median

        return new_weights
