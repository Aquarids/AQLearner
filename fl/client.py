import torch

class Client:
    def __init__(self, model: torch.nn.Module, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, train_loader, n_iters=10):
        self.model.train()
        for _ in range(n_iters):
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()

    def get_gradients(self):
        return [param.grad for param in self.model.parameters()]
    
    def update_model(self, new_model):
        self.model.load_state_dict(new_model.state_dict())