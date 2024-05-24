from fl.server import Server
from fl.model_metric import ModelMetric

import torch

type_fed_adagrad = 'fed_adagrad'
type_fed_yogi = 'fed_yogi'
type_fed_adam = 'fed_adam'


class FedAdaptOptServer(Server):

    def __init__(self, model, type, type_optimizer, lr=0.01, **kwargs):
        self.model = model
        self.type = type
        self.type_optimizer = type_optimizer
        self.lr = lr
        self.kwargs = kwargs
        self.model_metric = ModelMetric(type)
        self.vt = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
        }
        if type_optimizer == type_fed_adam:
            self.mt = {
                name: torch.zeros_like(param)
                for name, param in model.named_parameters()
            }

    def aggregate_weights(self, weights):
        if weights is None:
            return

        sum_deltas = {
            name: torch.zeros_like(param)
            for name, param in self.model.state_dict().items()
        }

        num_clients = len(weights)
        for client_weights in weights:
            for name, param in client_weights.items():
                sum_deltas[name] += self.model.state_dict()[name] - param
        avg_delta = {
            name: delta / num_clients
            for name, delta in sum_deltas.items()
        }

        if self.type_optimizer == type_fed_adagrad:
            self.fed_adagrad(avg_delta)
        elif self.type_optimizer == type_fed_yogi:
            self.fed_yogi(avg_delta)
        elif self.type_optimizer == type_fed_adam:
            self.fed_adam(avg_delta)

    def fed_adagrad(self, avg_delta):
        with torch.no_grad():
            for name, delta in avg_delta.items():
                self.vt[name] += delta**2
                update_step = -self.lr * delta / (torch.sqrt(self.vt[name]) +
                                                  self.kwargs['epsilon'])
                self.model.state_dict()[name].add_(update_step)

    def fed_yogi(self, avg_delta):
        with torch.no_grad():
            for name, delta in avg_delta.items():
                self.vt[name] += delta**2 - self.kwargs['lamb'] * (
                    torch.abs(self.vt[name] - delta**2) - self.kwargs['zeta'])
                update_step = -self.lr * delta / (torch.sqrt(self.vt[name]) +
                                                  self.kwargs['epsilon'])
                self.model.state_dict()[name].add_(update_step)

    def fed_adam(self, avg_delta):
        with torch.no_grad():
            for name, delta in avg_delta.items():
                self.mt[name] = self.kwargs['beta1'] * self.mt[name] + (
                    1 - self.kwargs['beta1']) * delta
                self.vt[name] = self.kwargs['beta2'] * self.vt[name] + (
                    1 - self.kwargs['beta2']) * delta**2
                update_step = -self.lr * self.mt[name] / (
                    torch.sqrt(self.vt[name]) + self.kwargs['epsilon'])
                self.model.state_dict()[name].add_(update_step)
