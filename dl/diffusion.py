import torch
from tqdm import tqdm


class DiffusionModel(torch.nn.Module):

    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.linear1 = torch.nn.Linear(784, 128)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128, 784)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten the input
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x.view(-1, 1, 28, 28)  # reshape the output to image size
        return x

    def fit(self, loader, learning_rate=0.01, n_epochs=1):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        progress_bar = tqdm(range(n_epochs * len(loader)),
                            desc="Training progress")
        for _ in range(n_epochs):
            for X, _ in loader:
                noise_level = 0.5
                noisy_data = self.corrupt(X, noise_level)
                reconstried_data = self.reverse_diffusion(noisy_data, steps=10)

                loss = criterion(reconstried_data, X)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.update(1)
        progress_bar.close()

    def corrupt(self, x, noise_level, mean=0, std=1):
        noise = torch.rand_like(x) * std * noise_level + mean
        return x + noise

    def reverse_diffusion(self, x, steps=10, std=1):
        for step in range(steps):
            noise_level = 1 - (step / steps)
            noise_reduction = std * noise_level
            x = x - torch.randn_like(x) * noise_reduction
            x = self.forward(x)
        return x
