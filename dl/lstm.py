import torch
from tqdm import tqdm

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        self.W_f = torch.nn.Parameter(torch.randn(input_size + hidden_size, hidden_size))
        self.b_f = torch.nn.Parameter(torch.zeros(hidden_size))

        self.W_i = torch.nn.Parameter(torch.randn(input_size + hidden_size, hidden_size))
        self.b_i = torch.nn.Parameter(torch.zeros(hidden_size))

        self.W_c = torch.nn.Parameter(torch.randn(input_size + hidden_size, hidden_size))
        self.b_c = torch.nn.Parameter(torch.zeros(hidden_size))

        self.W_o = torch.nn.Parameter(torch.randn(input_size + hidden_size, hidden_size))
        self.b_o = torch.nn.Parameter(torch.zeros(hidden_size))

        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h, c = torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)

        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat((x_t, h), dim=1)

            f = torch.sigmoid(combined @ self.W_f + self.b_f)
            i = torch.sigmoid(combined @ self.W_i + self.b_i)
            c_tilde = torch.tanh(combined @ self.W_c + self.b_c)
            o = torch.sigmoid(combined @ self.W_o + self.b_o)

            c = f * c + i * c_tilde
            h = o * torch.tanh(c)
        return h

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