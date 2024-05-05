import torch
from tqdm import tqdm

class SimpleCNNClassifier(torch.nn.Module):
    def __init__(self):
        super(SimpleCNNClassifier, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=3)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc = torch.nn.Linear(10 * 13 * 13, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 10 * 13 * 13)
        x = self.fc(x)
        return x
    
    def fit(self, loader, learning_rate=0.01, n_iters=10):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        progress_bar = tqdm(range(n_iters * len(loader)), desc="Training progress")
        for _ in range(n_iters):
            for X, y in loader:
                optimizer.zero_grad()
                outputs = self.forward(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                progress_bar.update(1)
        progress_bar.close()

    def predict(self, loader):
        self.eval()
        with torch.no_grad():
            predictions = []
            possibilities = []
            for X, _ in loader:
                possiblity = self.forward(X)
                possibilities += possiblity.tolist()
                predictions += torch.argmax(possiblity, dim=1).tolist()
            return predictions, possiblity
        
    def summary(self):
        print("Model Detail: ", self)        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params}")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}, Size: {param.size()}, Values: {param.data}")
