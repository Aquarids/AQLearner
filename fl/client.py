import torch
import torch.utils.data
from tqdm import tqdm

from fl.model_factory import type_regression, type_binary_classification, type_multi_classification

class Client:
    def __init__(self, model: torch.nn.Module, criterion, optimizer, type=type_regression):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.type = type
        self.train_loaders = None

    def setDataLoader(self, train_loader, n_iters=10):
        self.n_iters = n_iters
        self.train_loader = train_loader

    def train(self):
        self.model.train()

        progress_bar = tqdm(range(self.n_iters * len(self.train_loader)), desc="Client training progress")
        for _ in range(self.n_iters):
            for X_batch, y_batch in self.train_loader:
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

    def get_weights(self):
        return self.model.state_dict().copy()
    
    def update_model(self, state_dict):
        self.model.load_state_dict(state_dict)

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
                    predictions += torch.where(possiblity >= 0.5, 1, 0).tolist()
                elif type_regression == self.type:
                    possiblity = None
                    predictions += self.model(X).tolist()
            return predictions, possibilities