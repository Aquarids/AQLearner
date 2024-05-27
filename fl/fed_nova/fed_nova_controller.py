from fl.controller import FLController, mode_avg_weight
from fl.fed_nova.fed_nova_client import FedNovaClient
from fl.fed_nova.fed_nova_server import FedNovaServer

import torch
from tqdm import tqdm


class FedNovaController(FLController):

    def __init__(self, server: FedNovaServer, clients: list[FedNovaClient]):
        super().__init__(server, clients)

    def train(self, n_rounds, mode):
        if mode != mode_avg_weight:
            raise ValueError(f"Unsupported mode: {mode}")
        self.avg_weight_train(n_rounds)

    def avg_weight_train(self, n_rounds):
        self.server.model.train()
        progress_bar = tqdm(range(n_rounds * self.n_clients))
        for round_idx in range(n_rounds):
            updates = []

            for client_id in range(self.n_clients):
                progress_bar.set_description(
                    f"Avg weights training progress, round {round_idx + 1}, client {client_id + 1}"
                )
                client = self.clients[client_id]
                client.update_model(self.server.model.state_dict().copy())
                client.train(round_idx)

                updates.append(client.get_update())
                progress_bar.update(1)

            self.server.aggregate_updates(updates)
            self.server.eval(round_idx)
        progress_bar.close()
