from fl.controller import FLController, mode_avg_weight
from fl.scaffold.scaffold_server import ScaffoldServer
from fl.scaffold.scaffold_client import ScaffoldClient

import torch
from tqdm import tqdm


class ScaffoldController(FLController):

    def __init__(self, server: ScaffoldServer, clients: list[ScaffoldClient]):
        super().__init__(server, clients)

    def train(self, n_rounds, mode):
        if mode != mode_avg_weight:
            raise ValueError(f"Unsupported mode: {mode}")
        self.avg_weight_train(n_rounds)

    def avg_weight_train(self, n_rounds):
        self.server.model.train()
        progress_bar = tqdm(range(n_rounds * self.n_clients))
        for round_idx in range(n_rounds):
            weights = []
            c_clients = []

            for client_id in range(self.n_clients):
                progress_bar.set_description(
                    f"Avg weights training progress, round {round_idx + 1}, client {client_id + 1}"
                )
                client = self.clients[client_id]
                client.update_model(self.server.model.state_dict().copy())

                global_c = self.server.get_global_c()
                client.train(self.server.model, global_c, round_idx)
                weights.append(client.get_weights())
                c_clients.append(client.get_client_c())
                progress_bar.update(1)

            self.server.aggregate_weights(weights, c_clients)
            self.server.eval(round_idx)
        progress_bar.close()
