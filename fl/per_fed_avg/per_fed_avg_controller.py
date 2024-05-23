from fl.controller import FLController, mode_avg_weight
from fl.per_fed_avg.per_fed_avg_client import PerFedAvgClient

from tqdm import tqdm


class PerFedAvgController(FLController):

    def __init__(self, server, clients: list[PerFedAvgClient]):
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

            for client_id in range(self.n_clients):
                progress_bar.set_description(
                    f"Avg weights training progress, round {round_idx + 1}, client {client_id + 1}"
                )
                client = self.clients[client_id]
                client.update_model(self.server.model.state_dict().copy())
                client.train(round_idx)
                weights.append(client.get_weights())
                progress_bar.update(1)

            self.aggregate_weights(weights)
            self.server.eval(round_idx)

            # local personalization
            for client in self.clients:
                client.personalize()
        progress_bar.close()
