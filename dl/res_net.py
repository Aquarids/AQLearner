import torch
from tqdm import tqdm


class Residual(torch.nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = torch.nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = torch.nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = torch.nn.functional.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return torch.nn.functional.relu(Y)

class ResNet(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet, self).__init__()
        self.b1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
                   torch.nn.BatchNorm2d(32), torch.nn.ReLU(),
                   torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.b2 = torch.nn.Sequential(*self.resnet_block(32, 32, 2, first_block=True))
        self.b3 = torch.nn.Sequential(*self.resnet_block(32, 64, 2))

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, num_classes))

    def resnet_block(self, input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.fc(x)
        return y
    
    def fit(self, loader, learning_rate=0.01, n_iters=1000):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        progress_bar = tqdm(range(n_iters), desc="Training progress")
        for _ in progress_bar:
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
            for X, _ in loader:
                possiblity = self.forward(X)
                predictions += torch.argmax(possiblity, dim=1).tolist()
            return predictions, possiblity
        
    def summary(self):
        print("Model Detail: ", self)        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params}")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}, Size: {param.size()}, Values: {param.data}")
        
