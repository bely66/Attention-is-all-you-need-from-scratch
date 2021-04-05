from typing import ClassVar
import torch 
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert(self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, query, key, value, mask):
        N = value.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # N number of samples/batch_size, 
        # Key_len sentence size, 
        # heads and heads dim reshapes the embedding dimension
        value = value.reshape(N, value_len, self.heads, self.head_dim)
        key = key.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        # output of energy should be (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd, nkhd-->nhqk", [query, key])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        #output is softmax(q*kT/ (k**(1/2)) )V
        output = torch.einsum("nhql,nlhd-->nqhd", [attention, value]).reshape(N, query_len, self.heads*self.head_dim)
        out = self.fc_out(output)

        return out


