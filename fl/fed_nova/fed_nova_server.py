from fl.server import Server

import torch
from tqdm import tqdm


class FedNovaServer(Server):

    def __init__(self, model, criterion, optimizer, type):
        super().__init__(model, criterion, optimizer, type)

    def aggregate_updates(self, client_updates):
        aggr_weight = {
            k: torch.zeros_like(v)
            for k, v in self.model.state_dict().items()
        }
        total_n_iter_batch = 0

        for _, n_iter_batch in client_updates:
            total_n_iter_batch += n_iter_batch

        for client_update, n_iter_batch in client_updates:
            update_w = n_iter_batch / total_n_iter_batch
            for k, v in client_update.items():
                aggr_weight[k] += update_w * v

        self.model.load_state_dict(aggr_weight)
