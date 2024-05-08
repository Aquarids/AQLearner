from tqdm import tqdm

from fl.controller import FLController, mode_avg_grad, mode_avg_weight, mode_avg_vote
from fla.defend.mpc.mpc_server import MPCServer
from fla.defend.mpc.mpc_client import MPCClient


class MPCController(FLController):

    def __init__(self, server: MPCServer, clients: list[MPCClient]):
        super().__init__(server, clients)

    def aggregate_grads(self, grads, noise):
        self.server.aggretate_gradients(grads, noise)

    def aggregate_weights(self, weights, noise):
        self.server.aggregate_weights(weights, noise)

    def avg_grad_train(self, n_rounds):
        self.server.model.train()
        progress_bar = tqdm(range(n_rounds * self.n_clients))
        for round_idx in range(n_rounds):
            summed_grads = None
            summed_noise = None

            for client_id in range(self.n_clients):
                progress_bar.set_description(
                    f"Avg gradients training progress, round {round_idx + 1}, client {client_id + 1}"
                )
                client: MPCClient = self.clients[client_id]
                client.update_model(self.server.model.state_dict())
                client.train(round_idx)

                summed_grads, summed_noise = client.get_gradients(
                    summed_grads, summed_noise)
                progress_bar.update(1)

            self.aggregate_grads(summed_grads, summed_noise)
            self.server.eval(round_idx)
        progress_bar.close()

    def avg_weight_train(self, n_rounds):
        self.server.model.train()
        progress_bar = tqdm(range(n_rounds * self.n_clients))
        for round_idx in range(n_rounds):
            summed_weights = None
            summed_noise = None

            for client_id in range(self.n_clients):
                progress_bar.set_description(
                    f"Avg weights training progress, round {round_idx + 1}, client {client_id + 1}"
                )
                client = self.clients[client_id]
                client.update_model(self.server.model.state_dict().copy())
                client.train(round_idx)

                summed_weights, summed_noise = client.get_weights(
                    summed_weights, summed_noise)
                progress_bar.update(1)

            self.aggregate_weights(summed_weights, summed_noise)
            self.server.eval(round_idx)
        progress_bar.close()
