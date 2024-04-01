import torch
from tqdm import tqdm

class rnn(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(rnn, self).__init__()
        self.hidden_size = hidden_size

        self.W_xh = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hh = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_hy = torch.nn.Parameter(torch.randn(hidden_size, output_size))

        self.b_h = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_y = torch.nn.Parameter(torch.zeros(output_size))

        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        h = torch.zeros(batch_size, self.hidden_size)
        for t in range(seq_len):
            x_t = x[:, t, :]

            h = torch.tanh(x_t @ self.W_xh + h @ self.W_hh + self.b_h)
        
        y = self.fc(h)
        return y
    
    def fit(self, loader, learning_rate=0.01, n_epochs=100):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.train()
        progress_bar = tqdm(range(n_epochs), desc="Training progress")
        for _ in progress_bar:
            for X, y in loader:
                optimizer.zero_grad()
                outputs = self.forward(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            progress_bar.update(1)

    def predict(self, loader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for X, _ in loader:
                predictions += self.forward(X).tolist()
        return predictions
        
    def summary(self):
        print("Model Detail: ", self)        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params}")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}, Size: {param.size()}, Values: {param.data}")
        