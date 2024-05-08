import torch
import numpy as np

from fl.model_metric import ModelMetric
from fl.model_factory import type_regression, type_binary_classification, type_multi_classification


class Server:

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer,
                 criterion,
                 type=type_regression):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.type = type
        self.model_metric = ModelMetric(type)

    # for evaluating model after each round
    def setTestLoader(self, test_loader):
        self.test_loader = test_loader

    def calculate_gradients(self, grads):
        if grads is None or len(grads) == 0:
            return None

        avg_grad = [torch.zeros_like(param) for param in grads[0]]
        num_clients = len(grads)

        for client_grads in grads:
            if client_grads is None:
                continue
            avg_grad = [a + b for a, b in zip(avg_grad, client_grads)]

        avg_grad = [grad / num_clients for grad in avg_grad]
        return avg_grad

    def aggretate_gradients(self, grads):
        if grads is None:
            return

        grads = self.calculate_gradients(grads)
        self.model.train()
        self.optimizer.zero_grad()
        for param, grad in zip(self.model.parameters(), grads):
            param.grad = grad
        self.optimizer.step()

    def calculate_weights(self, weights):
        if weights is None or len(weights) == 0:
            return None

        avg_weights = {}
        num_clients = len(weights)

        for client_weights in weights:
            if client_weights is None:
                continue
            for key, value in client_weights.items():
                if key not in avg_weights:
                    avg_weights[key] = value.clone()
                else:
                    avg_weights[key] += value

        for key in avg_weights:
            avg_weights[key] /= num_clients

        return avg_weights

    def aggregate_weights(self, weights):
        if weights is None:
            return

        weights = self.calculate_weights(weights)
        if weights is None:
            return

        self.model.load_state_dict(weights, strict=False)

    def calculate_votes(self, client_results):
        if client_results is None or len(client_results) == 0:
            return None

        batch_size = len(client_results[0][0])

        aggregated_predictions = [0] * batch_size

        for i in range(batch_size):
            batch_preds = [
                client_result[i] for client_result in client_results
            ]

            if self.type == type_regression:
                aggregated_predictions[i] = np.mean(batch_preds)
            else:
                values, counts = np.unique(batch_preds, return_counts=True)
                max_index = np.argmax(counts)
                aggregated_predictions[i] = values[max_index]

        # in order to compare with test loader
        adjusted_predictions = [[pred] for pred in aggregated_predictions]

        return adjusted_predictions

    def aggregate_votes(self, votes, round_idx):
        if votes is None:
            return

        aggregated_predictions = self.calculate_votes(votes)
        y_test_value = []
        for _, y in self.test_loader:
            y_test_value += y.tolist()

        self.model_metric.update(y_test_value, aggregated_predictions, None,
                                 round_idx)

    def eval(self, round_idx):
        y_pred_value, y_prob_value = self.predict(self.test_loader)
        y_test_value = []
        for _, y in self.test_loader:
            y_test_value += y.tolist()
        self.model_metric.update(y_test_value, y_pred_value, y_prob_value,
                                 round_idx)

    def get_model(self):
        return self.model

    def predict(self, loader):
        self.model.eval()
        with torch.no_grad():
            predictions = []
            possibilities = []
            for X, _ in loader:
                if type_multi_classification == self.type:
                    possiblity = self.model(X)
                    possibilities += possiblity.tolist()
                    predictions += torch.argmax(possiblity, dim=1).tolist()
                elif type_binary_classification == self.type:
                    possiblity = self.model(X)
                    possibilities += possiblity.tolist()
                    predictions += torch.where(possiblity >= 0.5, 1,
                                               0).tolist()
                elif type_regression == self.type:
                    possiblity = None
                    predictions += self.model(X).tolist()
            return predictions, possibilities

    def summary(self):
        self.model_metric.summary()
        print("Global model: ", self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters: {total_params}")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(
                    f"Layer: {name}, Size: {param.size()}, Values: {param.data}"
                )
