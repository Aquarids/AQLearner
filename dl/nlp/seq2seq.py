import torch
from tqdm import tqdm

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(Embedding, self).__init__()
        self.embedding = torch.nn.Parameter(torch.randn(vocab_size, emb_dim))
    
    def forward(self, x):
        return torch.nn.functional.embedding(x, self.embedding)

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = Embedding(input_dim, emb_dim)
        self.rnn = torch.nn.LSTM(emb_dim, hidden_dim)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, cell) = self.rnn(x)
        return output, hidden, cell
    
class Decoder(torch.nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = Embedding(output_dim, emb_dim)
        self.rnn = torch.nn.LSTM(emb_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden, cell):
        x = self.embedding(x).unsqueeze(0)
        output, (hidden, cell) = self.rnn(x, (hidden, cell))
        x = self.fc(output.squeeze(0))
        return x, hidden, cell

class Seq2Seq(torch.nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(src_vocab_size, emb_dim, hidden_dim)
        self.decoder = Decoder(tgt_vocab_size, emb_dim, hidden_dim)

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

    def fit(self, loader, learning_rate=0.01, epochs=10):
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        self.train()

        progress_bar = tqdm(range(epochs), desc="Training progress")
        for _ in progress_bar:
            for X, y in loader:
                optimizer.zero_grad()
                src, tgt = X.transpose(0, 1), y.transpose(0, 1)
                output, hidden, cell = self.encoder(src)
                output_logits = torch.zeros(tgt.shape[0], tgt.shape[1], self.tgt_vocab_size).to(src.device)

                # 解码器的第一个输入是<sos>
                input = tgt[0,:]
                for t in range(1, tgt.shape[0]):
                    output, hidden, cell = self.decoder(input, hidden, cell)
                    output_logits[t] = output
                    input = tgt[t]
                loss = criterion(output_logits[1:].reshape(-1, self.tgt_vocab_size), tgt[1:].reshape(-1))
                loss.backward()
                optimizer.step()
            progress_bar.update(1)
        progress_bar.close()

    def predict(self, src_loader, max_length=100):
        self.eval()
        predictions = []
        with torch.no_grad():
            for src, _ in src_loader:
                src = src.transpose(0, 1)
                _, hidden, cell = self.encoder(src)
                outputs = torch.zeros(src.shape[1], max_length).to(src.device)
                input = torch.tensor([1], device=src.device)  # <sos> 的索引

                for t in range(max_length):
                    output, hidden, cell = self.decoder(input, hidden, cell)
                    top1 = output.argmax(1)
                    outputs[:, t] = top1
                    input = top1
                predictions.append(outputs)
        return predictions
    
    def summary(self):
        print("Model Detail: ", self)        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params}")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}, Size: {param.size()}, Values: {param.data}")