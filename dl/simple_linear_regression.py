import torch
from tqdm import tqdm


class SimpleLinearRegression(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(SimpleLinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

    def fit(self, X, y, learning_rate=0.01, n_iters=1000):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        progress_bar = tqdm(range(n_iters), desc="Training progress")
        for _ in progress_bar:
            optimizer.zero_grad()
            outputs = self.forward(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
        progress_bar.close()

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self.forward(X)

    def summary(self):
        print("Model Detail: ", self)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params}")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(
                    f"Layer: {name}, Size: {param.size()}, Values: {param.data}"
                )
