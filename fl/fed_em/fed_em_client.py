from fl.client import Client
from fl.model_factory import type_binary_classification, type_multi_classification, type_regression

import torch
from tqdm import tqdm


class FedEMClient(Client):

    def __init__(self, models: list[torch.nn.Module], criterion, type):
        self.criterion = criterion
        self.type = type
        self.train_loader = None
        self.models = models
        self.responsibility = None

    def _e_step(self):
        progress_bar = tqdm(range(len(self.train_loader)),
                            desc="E-step progress")
        self.responsibility = torch.ones(len(self.train_loader),
                                         len(self.models)) / len(self.models)
        for i, (X_batch, y_batch) in enumerate(self.train_loader):
            batch_losses = []
            for j, model in enumerate(self.models):
                output = model(X_batch)
                loss = self.criterion(output, y_batch)
                batch_losses.append(loss)

            batch_losses = torch.stack(batch_losses)
            max_loss = batch_losses.max()
            self.responsibility[i, :] = torch.exp(-batch_losses + max_loss)
            self.responsibility[i, :] = self.responsibility[
                i, :] / self.responsibility[i, :].sum()

            progress_bar.update(1)
        progress_bar.close()

    def _m_step(self):
        progress_bar = tqdm(range(len(self.models) * len(self.train_loader)),
                            desc="M-step progress")
        for j, model in enumerate(self.models):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            for i, (X_batch, y_batch) in enumerate(self.train_loader):
                optimizer.zero_grad()
                output = model(X_batch)
                loss = self.criterion(output, y_batch)

                weighted_loss = loss * self.responsibility[i, j].detach()
                weighted_loss.backward()
                optimizer.step()

                progress_bar.update(1)
        progress_bar.close()

    def train(self, round_idx=-1):
        self._e_step()
        self._m_step()

    def get_gradients(self):
        return None

    def get_weights(self):
        return None

    def predict(self, loader):
        for model in self.models:
            model.eval()
        with torch.no_grad():
            predictions = []
            possibilities = []
            for X, _ in loader:
                if self.type == type_multi_classification:
                    possiblity = torch.stack(
                        [model(X) for model in self.models]).mean(dim=0)
                    possibilities += possiblity.tolist()
                    predictions += torch.argmax(possiblity, dim=1).tolist()
                    print(predictions)
                elif self.type == type_binary_classification:
                    possiblity = torch.stack(
                        [model(X) for model in self.models]).mean(dim=0)
                    possibilities += possiblity.tolist()
                    predictions += torch.where(possiblity >= 0.5, 1,
                                               0).tolist()
                elif self.type == type_regression:
                    possiblity = None
                    predictions += torch.stack([
                        model(X) for model in self.models
                    ]).mean(dim=0).tolist()
        return predictions, possibilities

    def get_vote(self, loader):
        y_pred, _ = self.predict(loader)
        return y_pred

    def update_model(self, state_dict):
        for model in self.models:
            model.load_state_dict(state_dict)
