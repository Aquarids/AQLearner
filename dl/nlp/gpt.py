import torch
from tqdm import tqdm

class Block(torch.nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(Block, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = torch.nn.LayerNorm(embed_size)
        self.norm2 = torch.nn.LayerNorm(embed_size)

        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embed_size, forward_expansion * embed_size),
            torch.nn.ReLU(),
            torch.nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        attention = self.attention(query, key, value, attn_mask=mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        output = self.dropout(self.norm2(forward + x))
        return output
    
class GPT(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length):
        super(GPT, self).__init__()
        self.word_embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = torch.nn.Embedding(max_length, embed_size)

        self.layers = torch.nn.ModuleList([
            Block(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)
        ])

        self.fc = torch.nn.Linear(embed_size, vocab_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        N, sequence_length = x.shape
        positions = torch.arange(0, sequence_length).unsqueeze(0).repeat(N, 1)

        out = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return self.fc(out)
    
    def mask(self, size):
        mask = torch.tril(torch.ones(size, size)).type(torch.bool)
        return mask
    
    def train_epoch(self, loader, optimizer, criterion):
        for x, y in loader:
            optimizer.zero_grad()
            output = self.forward(x, self.mask(x.size(1)))
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

    def train(self, train_loader, val_loader, learning_rate=0.01, n_epochs=10):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        progress_bar = tqdm(range(n_epochs), desc="Training progress")
        for _ in range(n_epochs):
            self.train_epoch(train_loader, optimizer, criterion)
            val_accuracy = self.evaluate(val_loader)
            progress_bar.set_postfix(f"Validation accuracy: {val_accuracy:.4f}")
            progress_bar.update(1)
        progress_bar.close()

    def evaluate(self, loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                output = self.forward(x, self.mask(x.size(1)))
                _, predicted = torch.max(output, dim=2)
                correct += (predicted == y).sum().item()
                total += y.size(0) * y.size(1)
        return correct / total

    def predict(self, start_text, max_length, vocab):
        self.eval()
        with torch.no_grad():
            tokens = [vocab[token] for token in start_text.split()]
            for _ in range(max_length):
                x = torch.tensor(tokens).unsqueeze(0)
                output = self.forward(x, self.mask(x.size(1)))
                _, predicted = torch.max(output[:, -1, :], dim=1)
                tokens.append(predicted.item())
                if predicted.item() == vocab["<eos>"]:
                    break
        return " ".join([vocab[token] for token in tokens])