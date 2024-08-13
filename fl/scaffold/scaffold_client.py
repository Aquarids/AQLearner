from fl.client import Client

import torch
from tqdm import tqdm


class ScaffoldClient(Client):

    def __init__(self,
                 model: torch.nn.Module,
                 criterion,
                 optimizer,
                 type,
                 c_lr=0.01):
        super().__init__(model, criterion, optimizer, type)
        self.c = [torch.zeros_like(p) for p in model.parameters()]
        self.c_lr = c_lr

    def train(self, global_model: torch.nn.Module, global_c, round_idx=-1):
        self.model.train()

        progress_bar = tqdm(range(self.n_iters * len(self.train_loader)),
                            desc="Client training progress")
        for _ in range(self.n_iters):
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()

                with torch.no_grad():
                    for w_l, c_l, c_g in zip(self.model.parameters(), self.c,
                                             global_c):
                        w_l.grad += c_g - c_l

                self.optimizer.step()
                progress_bar.update(1)

            with torch.no_grad():
                for w_l, w_g, c_l in zip(self.model.parameters(),
                                         global_model.parameters(), self.c):
                    c_l += (w_l - w_g - c_l) * self.c_lr

        progress_bar.close()

    def get_client_c(self):
        return self.c
