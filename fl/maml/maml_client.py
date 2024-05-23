from fl.client import Client

import torch
import copy
from tqdm import tqdm


class MAMLClient(Client):

    def __init__(self, model: torch.nn.Module, criterion, optimizer, inner_lr,
                 type):
        super().__init__(model, criterion, optimizer, type)
        self.inner_lr = inner_lr

    def setDataLoader(self, tasks, n_iters=10):
        self.n_iters = n_iters
        self.tasks = tasks

    def train(self, round_idx=-1):

        progress_bar = tqdm(range(self.n_iters * len(self.tasks)),
                            desc="Client training progress")
        for _ in range(self.n_iters):
            self.optimizer.zero_grad()

            for task_train_loader, task_val_loader in self.tasks:

                adapted_model = copy.deepcopy(self.model)
                adapted_model.train()

                for X_batch, y_batch in task_train_loader:
                    inner_optimizer = torch.optim.SGD(
                        adapted_model.parameters(), lr=self.inner_lr)
                    inner_optimizer.zero_grad()
                    output = adapted_model(X_batch)
                    loss = self.criterion(output, y_batch)
                    loss.backward()
                    inner_optimizer.step()

                adapted_model.eval()
                with torch.no_grad():
                    for X_batch, y_batch in task_val_loader:
                        output = adapted_model(X_batch)
                        loss = self.criterion(output, y_batch)

                # use the last data to compute gradients
                adapted_model.train()
                output = adapted_model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()

                # transfer gradients to the original model
                for param, adapted_param in zip(self.model.parameters(),
                                                adapted_model.parameters()):
                    if adapted_param.grad is None:
                        continue
                    if param.grad is None:
                        param.grad = adapted_param.grad.clone()
                    else:
                        param.grad += adapted_param.grad.clone()
                progress_bar.update(1)

            self.optimizer.step()

        progress_bar.close()
