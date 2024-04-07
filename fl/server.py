import torch

from tqdm import tqdm
from fl.model_metric import ModelMetric, type_regresion, type_binary_classification, type_multi_classification

class Server:
    def __init__(self, model: torch.nn.Module, optimizer, criterion, type=type_regresion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.type = type
        self.model_metric = ModelMetric(type)

    # for evaluating model after each round
    def setTestLoader(self, test_loader):
        self.test_loader = test_loader

    def train(self, n_rounds, clients):
        self.model_metric.reset()
        n_clients = len(clients)
        progress_bar = tqdm(range(n_rounds * n_clients), desc="Training progress")
        for round_idx in range(n_rounds):
            gradients = []
            for client_id in range(n_clients):
                client = clients[client_id]
                client.train(round_idx)
                gradients.append(client.get_gradients())
                progress_bar.update(1)
            self.aggretate_gradients(gradients)

            y_pred_value, y_prob_value = self.predict(self.test_loader)
            y_test_value = []
            for _, y in self.test_loader:
                y_test_value += y.tolist()
            self.model_metric.update(y_test_value, y_pred_value, y_prob_value, round_idx)

        progress_bar.close()
        
    def aggretate_gradients(self, gradients):
        if gradients is None:
            return
        
        n_clients = len(gradients)

        avg_grad = []
        for grads in zip(*gradients):
            avg_grad.append(torch.stack(grads).sum(0) / n_clients)
        
        for param, grad in zip(self.model.parameters(), avg_grad):
            param.grad = grad

        self.model.train()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_model(self):
        return self.model
    
    def predict(self, loader):
        self.model.eval()
        with torch.no_grad():
            predictions = []
            possiblities = []
            for X, _ in loader:
                if type_multi_classification == self.type:
                    possiblity = self.model(X)
                    possiblities += possiblity.tolist()
                    predictions += torch.argmax(possiblity, dim=1).tolist()
                elif type_binary_classification == self.type:
                    possiblity = self.model(X)
                    possiblities += possiblity.tolist()
                    predictions += torch.where(possiblity >= 0.5, 1, 0).tolist()
                elif type_regresion == self.type:
                    possiblity = None
                    predictions += self.model(X).tolist()
            return predictions, possiblities
        
    def summary(self):
        self.model_metric.summary()
        print("Global model: ", self.model)        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters: {total_params}")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}, Size: {param.size()}, Values: {param.data}")
