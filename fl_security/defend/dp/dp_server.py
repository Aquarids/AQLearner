import torch
from torch.nn.modules import Module

from fl.model_factory import type_regression
from fl.server import Server


class OutputPerturbationServer(Server):

    def __init__(self,
                 model: Module,
                 optimizer,
                 criterion,
                 type=type_regression):
        super().__init__(model, optimizer, criterion, type)

    def predict(self, loader, epsilon=0.1):
        y_pred, y_prob = super().predict(loader)
        y_pred = torch.tensor(y_pred)
        y_pred += torch.randn_like(y_pred) * epsilon
        if y_prob is not None:
            y_prob = torch.tensor(y_prob)
            y_prob += torch.randn_like(y_prob) * epsilon
        return y_pred.tolist(), y_prob.tolist()
