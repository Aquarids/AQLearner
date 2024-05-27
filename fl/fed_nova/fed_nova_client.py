from fl.client import Client

import torch
from tqdm import tqdm


class FedNovaClient(Client):

    def __init__(self, model: torch.nn.Module, criterion, optimizer, type):
        super().__init__(model, criterion, optimizer, type)

    def reset(self):
        self.n_iter_batch = 0
        self.update = {
            k: torch.zeros_like(v)
            for k, v in self.model.state_dict().items()
        }

    def train(self, round_idx=-1):
        self.model.train()
        self.reset()

        progress_bar = tqdm(range(self.n_iters * len(self.train_loader)),
                            desc="Client training progress")
        for _ in range(self.n_iters):
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()

                for k, v in self.model.state_dict().items():
                    self.update[k] += v
                self.n_iter_batch += 1

                progress_bar.update(1)

        for k, _ in self.update.items():
            self.update[k] /= self.n_iter_batch

        progress_bar.close()

    def get_update(self):
        return (self.update, self.n_iter_batch)
