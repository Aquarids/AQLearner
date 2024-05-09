import torch
import math


class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attention, value)
        return output


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size needs to be divisible by heads"

        self.values = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = torch.nn.Linear(self.head_dim,
                                       self.head_dim,
                                       bias=False)
        self.fc_out = torch.nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, query, values, keys, mask=None):

        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[
            1], query.shape[1]

        values = values.reshape(N, self.heads, value_len, self.head_dim)
        keys = keys.reshape(N, self.heads, key_len, self.head_dim)
        queries = query.reshape(N, self.heads, query_len, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        attention = ScaledDotProductAttention()(queries, keys, values, mask)
        attention = attention.permute(0, 2, 1, 3).contiguous()
        attention = attention.reshape(N, query_len, self.heads * self.head_dim)

        return self.fc_out(attention)


class SelfAttention(torch.nn.Module):

    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embed_size, heads)
        self.norm = torch.nn.LayerNorm(embed_size)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, value, key, query, mask=None):
        attention = self.multi_head_attention(value, key, query, mask)
        x = self.dropout(self.norm(attention + query))
        return x
