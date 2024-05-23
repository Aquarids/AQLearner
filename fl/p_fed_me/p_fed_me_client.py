from fl.client import Client

import torch
from tqdm import tqdm


class pFedMeClient(Client):

    def __init__(self, model: torch.nn.Module, criterion, optimizer, type,
                 beta):
        super().__init__(model, criterion, optimizer, type)
        self.beta = beta

    def train(self, global_model: torch.nn.Module, round_idx=-1):
        self.model.train()

        progress_bar = tqdm(range(self.n_iters * len(self.train_loader)),
                            desc="Client training progress")
        for _ in range(self.n_iters):
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)

                reg_term = 0
                if self.beta is not None:
                    for w0, w1 in zip(self.model.parameters(),
                                      global_model.parameters()):
                        reg_term += (w0 - w1).norm()**2
                    reg_term = self.beta * reg_term
                loss += reg_term

                loss.backward()
                self.optimizer.step()
                progress_bar.update(1)
        progress_bar.close()
