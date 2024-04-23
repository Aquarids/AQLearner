import torch
from tqdm import tqdm
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class TinyYolo(torch.nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20):
        super(TinyYolo, self).__init__()

        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = torch.nn.Linear(32 * 50 * 50, grid_size * grid_size * (num_classes + 5 * num_boxes))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.view(-1, self.grid_size, self.grid_size, (self.num_classes + 5 * self.num_boxes))
    
    def fit(self, loader, learning_rate=0.01, n_iters=1000):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        progress_bar = tqdm(range(n_iters * len(loader)), desc="Training progress")
        for _ in range(n_iters):
            for images, targets in loader:
                optimizer.zero_grad()
                output = self.forward(images)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                progress_bar.update(1)
        progress_bar.close()

    def predict(self, image):
        self.eval()
        with torch.no_grad():
            prediction = self.forward(image.unsqueeze(0))
        fig, ax = plt.subplots(1)
        ax.imshow(image.permute(1, 2, 0).numpy())

        for box in prediction.squeeze(0):
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.show()

