import torch

type_regresion = "regression"
type_classification = "classification"

class Server:
    def __init__(self, model: torch.nn.Module, optimizer, criterion, type=type_regresion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.type = type
        
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
            for X, _ in loader:
                if type_classification == self.type:
                    possiblity = self.model(X)
                    predictions += torch.argmax(possiblity, dim=1).tolist()
                elif type_regresion == self.type:
                    possiblity = None
                    predictions += self.model(X).tolist()
            return predictions, possiblity
        
    def summary(self):
        print("Global model: ", self.model)        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters: {total_params}")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}, Size: {param.size()}, Values: {param.data}")
