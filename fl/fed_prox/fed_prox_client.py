from fl.client import Client

import torch
from tqdm import tqdm


class FedProxClient(Client):

    def __init__(self, model: torch.nn.Module, criterion, optimizer, type, mu):
        super().__init__(model, criterion, optimizer, type)
        self.mu = mu

    def train(self, global_model: torch.nn.Module, round_idx=-1):
        self.model.train()

        progress_bar = tqdm(range(self.n_iters * len(self.train_loader)),
                            desc="Client training progress")
        for _ in range(self.n_iters):
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(X_batch)

                prox_term = 0
                if self.mu is not None:
                    prox_term = 0.5 * self.mu * sum([
                        (w - w0).norm()**2 for w, w0 in zip(
                            self.model.parameters(), global_model.parameters())
                    ])

                loss = self.criterion(output, y_batch) + prox_term
                loss.backward()
                self.optimizer.step()
                progress_bar.update(1)
        progress_bar.close()
