import torch
from torch import nn
from tqdm import tqdm

class CIFAR10Net(torch.nn.Module):
    # def __init__(self, n_classes=10):
    #     super(CIFAR10Net, self).__init__()
    #     self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
    #     self.pool1 = torch.nn.MaxPool2d(2, 2)

    #     self.conv2 = torch.nn.Conv2d(32, 64, 3, 1, 1)
    #     self.pool2 = torch.nn.MaxPool2d(2, 2)

    #     self.conv3 = torch.nn.Conv2d(64, 128, 3, 1, 1)
    #     self.pool3 = torch.nn.MaxPool2d(2, 2)

    #     self.fc1 = torch.nn.Linear(128 * 4 * 4, 256)
    #     self.fc2 = torch.nn.Linear(256, n_classes)

    #     self.flatten = torch.nn.Flatten()
    #     self.relu = torch.nn.ReLU()

    # def forward(self, x):
    #     x = self.pool1(self.relu(self.conv1(x)))
    #     x = self.pool2(self.relu(self.conv2(x)))
    #     x = self.pool3(self.relu(self.conv3(x)))
    #     x = self.flatten(x)
    #     x = self.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x

    def __init__(self, num_classes=10):
        super(CIFAR10Net, self).__init__()
        kernel_size = (3, 3)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def fit(self, loader, learning_rate=0.01, n_iters=10):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        progress_bar = tqdm(range(n_iters * len(loader)),
                            desc="Training progress")
        for _ in range(n_iters):
            for X, y in loader:
                optimizer.zero_grad()
                outputs = self.forward(X)
                loss = criterion(outputs, y)
                loss.backward()

                optimizer.step()
                progress_bar.set_postfix(loss=loss.item())
                progress_bar.update(1)
        progress_bar.close()

    def get_gradients(self):
        return self.grads

    def test(self, loader):
        self.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for X, y in loader:
                possibility = self.forward(X)
                preds = torch.argmax(possibility, dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)
        accuracy = total_correct / total_samples
        return accuracy

    def predict(self, loader):
        self.eval()
        with torch.no_grad():
            predictions = []
            possibilities = []
            for X, _ in loader:
                possibility = self.forward(X)
                possibilities += possibility.tolist()
                predictions += torch.argmax(possibility, dim=1).tolist()
            return predictions, possibility

    def summary(self):
        print("Model Detail: ", self)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params}")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(
                    f"Layer: {name}, Size: {param.size()}, Values: {param.data}"
                )