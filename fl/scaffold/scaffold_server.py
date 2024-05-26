from fl.server import Server

import torch


class ScaffoldServer(Server):

    def __init__(self, model, optimizer, criterion, type):
        super().__init__(model, optimizer, criterion, type)
        self.c = [torch.zeros_like(p) for p in self.model.parameters()]

    def aggregate_weights(self, weights, c_clients):
        if weights is None or len(weights) == 0:
            return None

        aggr_weights = {
            k: torch.zeros_like(v)
            for k, v in self.model.state_dict().items()
        }
        aggr_c = [torch.zeros_like(p) for p in self.model.parameters()]

        for client_weights, client_c in zip(weights, c_clients):
            for key, value in client_weights.items():
                aggr_weights[key] += value
            for c_g, c_l in zip(aggr_c, client_c):
                c_g += c_l

        n_clients = len(weights)
        for key in aggr_weights:
            aggr_weights[key] /= n_clients
        for c_g in aggr_c:
            c_g /= n_clients

        self.model.load_state_dict(aggr_weights)
        self.c = aggr_c

    def get_global_c(self):
        return self.c
