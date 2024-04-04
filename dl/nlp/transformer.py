import torch 
import dl.nlp.preprocess as Preprocess

from tqdm import tqdm
from dl.nlp.attention import MultiHeadAttention
from dl.nlp.position_encoding import PositionalEncoding

class FeedForward(torch.nn.Module):
    def __init__(self, d_model, ff_hidden_dim, dropout = 0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = torch.nn.Linear(d_model, ff_hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear_2 = torch.nn.Linear(ff_hidden_dim, d_model)
    
    def forward(self, x):
        x = torch.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_hidden_dim)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attention))
        forward = self.ff(x)
        x = self.norm2(x + self.dropout(forward))
        return x
    
class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout = 0.1):
        super(DecoderLayer, self).__init__()
        self.masked_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_hidden_dim)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        attention = self.masked_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attention))
        attention2 = self.attention(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attention2))
        forward = self.ff(x)
        x = self.norm3(x + self.dropout(forward))
        return x
    
class Encoder(torch.nn.Module):
    def __init__(self, src_vocab_size, d_model, num_layers, num_heads, ff_hidden_dim, dropout = 0.1):
        super(Encoder, self).__init__()
        self.embed = torch.nn.Embedding(src_vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model)
        self.layers = torch.nn.ModuleList([EncoderLayer(d_model, num_heads, ff_hidden_dim, dropout) for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return x
    
class Decoder(torch.nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_layers, num_heads, ff_hidden_dim, dropout = 0.1):
        super(Decoder, self).__init__()
        self.embed = torch.nn.Embedding(tgt_vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model)
        self.layers = torch.nn.ModuleList([DecoderLayer(d_model, num_heads, ff_hidden_dim, dropout) for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.embed(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x
    

class Transformer(torch.nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers, num_heads, ff_hidden_dim, dropout = 0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, ff_hidden_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, ff_hidden_dim, dropout)
        self.out = torch.nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        out = self.out(dec_out)
        return out
    
    def _get_mask(self, src, tgt, padding_index):
        tgt_seq_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len)).unsqueeze(0).unsqueeze(0)

        src_mask = (src != padding_index).unsqueeze(-2).unsqueeze(1)
        return src_mask, tgt_mask
    
    def fit(self, loader, learning_rate=0.01, epochs=10):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        progress_bar = tqdm(range(epochs), desc="Training progress")
        for _ in range(epochs):
            for src, tgt in loader:
                tgt_input = tgt[:, :-1]
                tgt_labels = tgt[:, 1:]

                src_mask, tgt_mask = self._get_mask(src, tgt_input, padding_index=Preprocess.pad_token)

                optimizer.zero_grad()
                outputs = self.forward(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_labels.contiguous().view(-1))
                loss.backward()
                optimizer.step()
            progress_bar.update(1)
        progress_bar.close()

    def predict(self, loader, max_len=50):
        self.eval()
        predictions = []

        with torch.no_grad():
            for src, _ in loader:
                src_mask = (src != 0).unsqueeze(-2) 
                curr_batch_size = src.size(0)
                decoder_input = torch.full((curr_batch_size, 1), fill_value=Preprocess.sos_token, dtype=torch.long)

                for _ in range(max_len):
                    tgt_mask = (torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1)), device=src.device)) == 1).transpose(1, 2)
                    output = self.forward(src, decoder_input, src_mask, tgt_mask)
                    next_token = output.argmax(dim=-1)[:, -1].unsqueeze(1)
                    decoder_input = torch.cat([decoder_input, next_token], dim=1) 

                    tokens_list = next_token.squeeze().tolist()
                    if not isinstance(tokens_list, list):
                        tokens_list = [tokens_list]

                    tokens_eos_check = [token == Preprocess.eos_token for token in tokens_list]
                    if all(tokens_eos_check):
                        break

                predictions.append(decoder_input)
        return predictions
        
    def summary(self):
        print("Model Detail: ", self)        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params}")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}, Size: {param.size()}, Values: {param.data}")