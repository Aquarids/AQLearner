import torch

from fl.server import Server
from fl_security.defend.mpc.mpc_decryptor import MPCDecryptor


class MPCServer(Server):

    def __init__(self, model: torch.nn.Module, optimizer, criterion, type,
                 n_clients):
        super().__init__(model, optimizer, criterion, type)
        self.n_clients = n_clients
        self.decryptor = MPCDecryptor()

    def calculate_gradients(self, grads, noise):
        decrypted_sum_grads = self.decryptor.decrypt_summed_gradients(
            grads, noise)
        avg_grads = []
        for grad in decrypted_sum_grads:
            avg_grads.append(grad / self.n_clients)
        return avg_grads

    def aggretate_gradients(self, grads, noise):
        if grads is None:
            return

        grads = self.calculate_gradients(grads, noise)

        self.model.train()
        self.optimizer.zero_grad()
        for param, grad in zip(self.model.parameters(), grads):
            param.grad = grad
        self.optimizer.step()

    def calculate_weights(self, weights, noise):
        decrypted_weights = self.decryptor.decrypt_weights(weights, noise)
        avg_weights = {}
        num_clients = len(weights)

        for key, value in decrypted_weights.items():
            if key not in avg_weights:
                avg_weights[key] = value.clone()
            else:
                avg_weights[key] += value

        for key in avg_weights:
            avg_weights[key] /= num_clients

        return avg_weights

    def aggregate_weights(self, weights, noise):
        avg_weights = self.calculate_weights(weights, noise)
        self.model.load_state_dict(avg_weights)

    def summary(self):
        print(f"Max diff of gradients: {self.decryptor.max_diff}")
        super().summary()
