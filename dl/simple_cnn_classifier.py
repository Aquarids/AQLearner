import torch
from tqdm import tqdm


class SimpleCNNClassifier(torch.nn.Module):

    def __init__(self):
        super(SimpleCNNClassifier, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1, 1)
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(2, 2, 0)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1, 1)
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(2, 2, 0)
        self.linear1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.linear2 = torch.nn.Linear(128, 10)
        self.softmax = torch.nn.Softmax(dim=1)

        self.losses = []
        self.grads = []

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = output.view(-1, 64 * 7 * 7)
        output = self.linear1(output)
        output = self.linear2(output)
        output = self.softmax(output)
        return output

    def fit(self, loader, learning_rate=0.001, n_iters=10):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        progress_bar = tqdm(range(n_iters * len(loader)),
                            desc="Training progress")
        for _ in range(n_iters):
            for X, y in loader:
                optimizer.zero_grad()
                outputs = self.forward(X)
                loss = criterion(outputs, y)
                loss.backward()

                self.grads.append([param.grad.clone() for param in self.parameters()])

                optimizer.step()
                progress_bar.set_postfix(loss=loss.item())
                progress_bar.update(1)
        progress_bar.close()

    def get_gradients(self):
        return self.grads

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
                print(
                    f"Layer: {name}, Size: {param.size()}, Values: {param.data}"
                )
