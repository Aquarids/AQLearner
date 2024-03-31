import torch
from tqdm import tqdm

class SimpleLogisticRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def _forward(self, x):
        return torch.sigmoid(self.linear(x))
    
    def fit(self, X, y, learning_rate=0.01, n_iters=1000):
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        progress_bar = tqdm(range(n_iters), desc="Training progress")
        for _ in progress_bar:
            optimizer.zero_grad()
            outputs = self._forward(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
        progress_bar.close()

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            p = self._forward(X)
            return torch.where(p >= 0.5, 1, 0)
        
    def summary(self):
        print("Model Detail: ", self)        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params}")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}, Size: {param.size()}, Values: {param.data}")