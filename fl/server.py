import torch

from fl.model_metric import ModelMetric
from fl.model_factory import type_regresion, type_binary_classification, type_multi_classification

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
        
    def aggretate_gradients(self, grads):
        if grads is None:
            return

        avg_grad = []
        for grad in grads:
            if grad is None:
                continue
            if len(avg_grad) == 0:
                avg_grad = grad
            else:
                avg_grad = [a + b for a, b in zip(avg_grad, grad)]
        
        for param, grad in zip(self.model.parameters(), avg_grad):
            param.grad = grad

        self.model.train()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def eval(self, round_idx):
        y_pred_value, y_prob_value = self.predict(self.test_loader)
        y_test_value = []
        for _, y in self.test_loader:
            y_test_value += y.tolist()
        self.model_metric.update(y_test_value, y_pred_value, y_prob_value, round_idx)

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
