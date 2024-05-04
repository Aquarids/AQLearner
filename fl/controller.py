from tqdm import tqdm
from fl.client import Client
from fl.server import Server

class FLController:
    def __init__(self, server: Server, clients: list[Client]):
        if len(clients) == 0:
            raise ValueError("Clients can not be empty")
        self.server = server
        self.clients = clients
        self.n_clients = len(clients)

    def train(self, n_rounds):
        progress_bar = tqdm(range(n_rounds * self.n_clients))
        for round_idx in range(n_rounds):
            gradients = []

            for client_id in range(self.n_clients):
                progress_bar.set_description(f"Training progress, round {round_idx}, client {client_id}")
                client = self.clients[client_id]
                client.train()
                gradients.append(client.get_gradients())
                
                progress_bar.update(1)
            self.server.aggretate_gradients(gradients)
            self.server.eval(round_idx)
        progress_bar.close()
        