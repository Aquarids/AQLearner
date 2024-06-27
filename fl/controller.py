from tqdm import tqdm
from fl.client import Client
from fl.server import Server
import torch

mode_avg_grad = "avg_grad"
mode_avg_weight = "avg_weight"
mode_avg_vote = "avg_vote"


# assume controller as net to control the training process
class FLController:

    def __init__(self, server: Server, clients: list[Client]):
        if len(clients) == 0:
            raise ValueError("Clients can not be empty")
        self.server = server
        self.clients = clients
        self.n_clients = len(clients)

    def aggregate_grads(self, grads):
        self.server.aggretate_gradients(grads)

    def aggregate_weights(self, weights):
        self.server.aggregate_weights(weights)

    def aggregate_votes(self, votes, round_idx):
        self.server.aggregate_votes(votes, round_idx)

    def train(self, n_rounds, mode=mode_avg_grad):
        if mode == mode_avg_grad:
            self.avg_grad_train(n_rounds)
        elif mode == mode_avg_weight:
            self.avg_weight_train(n_rounds)
        elif mode == mode_avg_vote:
            self.avg_vote_train(n_rounds)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def client_train(self, client: Client, round_idx):
        client.update_model(self.server.model.state_dict())
        client.train(round_idx)

    def predict(self, loader):
        return self.server.predict(loader)

    def avg_grad_train(self, n_rounds):
        progress_bar = tqdm(range(n_rounds * self.n_clients))
        for round_idx in range(n_rounds):
            gradients = []

            for client_id in range(self.n_clients):
                progress_bar.set_description(
                    f"Avg gradients training progress, round {round_idx + 1}, client {client_id + 1}"
                )
                client = self.clients[client_id]
                self.client_train(client, round_idx)
                gradients.append(client.get_gradients())
                progress_bar.update(1)

            self.aggregate_grads(gradients)

            self.server.eval(round_idx)
        progress_bar.close()

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
        progress_bar.close()

    def avg_vote_train(self, n_rounds):
        # no need to train the server, just train the clients
        progress_bar = tqdm(range(n_rounds * self.n_clients))
        for round_idx in range(n_rounds):
            for client_id in range(self.n_clients):
                progress_bar.set_description(
                    f"Avg vote training progress, round {round_idx + 1}, client {client_id + 1}"
                )
                client = self.clients[client_id]
                client.train(round_idx)
                progress_bar.update(1)

            test_loader = self.server.test_loader
            client_result = []
            for client in self.clients:
                client_result.append(client.get_vote(test_loader))
            self.aggregate_votes(client_result, round_idx)
        progress_bar.close()
