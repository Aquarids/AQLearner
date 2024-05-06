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

        aligned_grads = [list(param_grads) for param_grads in zip(*grads)]
        grads = [param.grad for param in self.model.parameters()]

        median_grads = []
        for param_group in aligned_grads:
            stacked_grads = torch.stack(param_group)
            median = torch.median(stacked_grads, dim=0).values
            median_grads.append(median)
        
        return median_grads

    def trimmed_mean_aggr(self, grads, trim_ratio=0.1):
        if grads is None or len(grads) == 0:
            return None
        
        num_parameters = len(grads[0])
        num_clients = len(grads)
        num_trim = int(num_clients * trim_ratio)

        if num_trim * 2 >= num_clients:
            raise ValueError("Trim ratio too high for the number of clients")

        aggregated_gradients = []
        for i in range(num_parameters):
            param_grads = torch.stack([client_grads[i] for client_grads in grads])
            flattened_grads = param_grads.view(param_grads.shape[0], -1)
            norms = torch.norm(flattened_grads, dim=1)
            sorted_indices = torch.argsort(norms)

            valid_indices = sorted_indices[num_trim:-num_trim] if num_trim > 0 else sorted_indices

            valid_grads = flattened_grads[valid_indices]
            trimmed_mean = valid_grads.mean(dim=0)

            trimmed_mean_reshaped = trimmed_mean.view(*param_grads.shape[1:])
            aggregated_gradients.append(trimmed_mean_reshaped)

        return aggregated_gradients