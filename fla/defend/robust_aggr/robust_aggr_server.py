from fl.server import Server

import torch
import numpy as np

type_aggr_avg = "avg"
type_aggr_median = "median"
type_aggr_trimmed_mean = "trimmed_mean"
type_aggr_krum = "krum"

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
            return self.grad_median_aggr(grads)
        elif type_aggr == type_aggr_trimmed_mean:
            return self.grad_trimmed_mean_aggr(grads, kwargs["trim_ratio"])
        else:
            return super().calculate_gradients(grads)
    
    def grad_median_aggr(self, grads):
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

    def grad_trimmed_mean_aggr(self, grads, trim_ratio=0.1):
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
    
    def aggregate_weights(self, weights, type_aggr, **kwargs):
        new_weights = self.calculate_weights(weights, type_aggr, **kwargs)
        if new_weights is None:
            return
        self.model.load_state_dict(new_weights, strict=False)
        
    def calculate_weights(self, weights, type_aggr, **kwargs):
        if type_aggr == type_aggr_median:
            return self.weight_median_aggr(weights)
        elif type_aggr == type_aggr_trimmed_mean:
            return self.weight_trimmed_mean_aggr(weights, kwargs["trim_ratio"])
        elif type_aggr == type_aggr_krum:
            return self.weight_krum_aggr(weights, kwargs["n_malicious"])
        else:
            return super().calculate_weights(weights)
        
    def weight_median_aggr(self, weights):
        if weights is None or len(weights) == 0:
            return None

        new_weights = {}

        if len(weights) > 0 and isinstance(weights[0], dict):
            keys = weights[0].keys()
        else:
            return None

        for key in keys:
            param_group = [client_weights[key] for client_weights in weights if key in client_weights]
            stacked_weights = torch.stack(param_group)
            median = torch.median(stacked_weights, dim=0).values
            new_weights[key] = median

        return new_weights
    
    def weight_trimmed_mean_aggr(self, weights, trim_ratio=0.1):
        if weights is None or len(weights) == 0:
            return None

        num_clients = len(weights)
        num_trim = int(num_clients * trim_ratio)

        if num_trim * 2 >= num_clients:
            raise ValueError("Trim ratio too high for the number of clients")

        align_weights = {}
        for client_weights in weights:
            if client_weights is None:
                continue
            for key, value in client_weights.items():
                if key not in align_weights:
                    align_weights[key] = []
                align_weights[key].append(value)
        
        new_weights = {}
        for key, param_group in align_weights.items():
            stacked_weights = torch.stack(param_group)
            flattened_weights = stacked_weights.view(stacked_weights.shape[0], -1)
            norms = torch.norm(flattened_weights, dim=1)
            sorted_indices = torch.argsort(norms)

            valid_indices = sorted_indices[num_trim:-num_trim] if num_trim > 0 else sorted_indices

            valid_weights = flattened_weights[valid_indices]
            trimmed_mean = valid_weights.mean(dim=0)

            trimmed_mean_reshaped = trimmed_mean.view(*stacked_weights.shape[1:])
            new_weights[key] = trimmed_mean_reshaped
        
        return new_weights
    
    def weight_krum_aggr(self, weights, n_malicious=1):
        scores = []
        n_clients = len(weights)

        for i in range(n_clients):
            distances = []
            for j in range(n_clients):
                if i == j:
                    continue
                distance = np.linalg.norm(weights[i] - weights[j])
                distances.append(distance)

            distances.sort()
            krum_score = sum(distances[:n_clients - n_malicious - 2])
            scores.append(krum_score)

        chosen_index = np.argmin(scores)
        return weights[chosen_index]


