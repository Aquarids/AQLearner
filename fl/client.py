import torch
import torch.utils.data
from tqdm import tqdm

class Client:
    def __init__(self, model: torch.nn.Module, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loaders = []

    def setDataLoader(self, train_loader, n_iters=10):
        self.n_iters = n_iters
        self.train_loader = train_loader

    def train(self):
        self.model.train()
        train_loader = self.train_loader

        progress_bar = tqdm(range(self.n_iters * len(train_loader)), desc="Client training progress")
        for _ in range(self.n_iters):
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
                progress_bar.update(1)
        progress_bar.close()
    
    def get_gradients(self):
        grads = [param.grad for param in self.model.parameters()]
        return grads
    
    def update_model(self, new_model):
        self.model.load_state_dict(new_model.state_dict())