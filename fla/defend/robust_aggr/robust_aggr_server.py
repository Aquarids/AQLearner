from fl.server import Server

import torch

type_aggr_median = "median"
type_aggr_trimmed_mean = "trimmed_mean"

class RobustAggrServer(Server):
    def __init__(self, model, optimizer, criterion, type):
        super().__init__(model, optimizer, criterion, type)

    def aggretate_gradients(self, grads, type_aggr, **kwargs):
        aggr_grads = self.calculate_gradients(grads, type_aggr, **kwargs)
        for param, grad in zip(self.model.parameters(), aggr_grads):
            param.grad = grad
        self.optimizer.step()
        self.optimizer.zero_grad()

    def calculate_gradients(self, grads, type_aggr, **kwargs):
        if type_aggr == type_aggr_median:
            return self.median_aggr(grads)
        elif type_aggr == type_aggr_trimmed_mean:
            return self.trimmed_mean_aggr(grads, **kwargs)
        else:
            return super().calculate_gradients(grads)
    
    def median_aggr(self, grads):
        if grads is None or len(grads) == 0:
            return None

        num_params = len(grads[0])
        median_grads = [None] * num_params

        for param_idx in range(num_params):
            param_grads = torch.stack([client_grads[param_idx] for client_grads in grads])
            median_grad = torch.median(param_grads, dim=0)[0]
            median_grads[param_idx] = median_grad

        return median_grads

    def trimmed_mean_aggr(self, grads, trim_ratio=0.1):
        if not grads:
            return None

        num_params = len(grads[0])
        num_clients = len(grads)
        num_trim = int(num_clients * trim_ratio)

        trimmed_mean_grads = [None] * num_params

        for param_idx in range(num_params):
            param_grads = torch.stack([client_grads[param_idx] for client_grads in grads])

            flattened_grads = param_grads.view(param_grads.shape[0], -1)

            norms = torch.norm(flattened_grads, dim=1)
            sorted_indices = torch.argsort(norms)

            if len(sorted_indices) > 2 * num_trim:
                valid_indices = sorted_indices[num_trim:-num_trim]
            else:
                valid_indices = sorted_indices

            trimmed_grads = flattened_grads[valid_indices].mean(dim=0)

            trimmed_mean_grads[param_idx] = trimmed_grads.view(*param_grads.shape[1:])

        return trimmed_mean_grads