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

    def forward(self, query, key, value, mask=None):
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

class Transformer(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(Tranformer, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm_1 = nn.LayerNorm(embed_size)
        self.norm_2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        attention = self.attention(query, key, value, mask)
        norm_1 = self.dropout(self.norm_1(attention + query))
        forward = self.feed_forward(norm_1)
        out = self.dropout(self.norm_2(forward + norm_1))
        return out

class Encoder(nn.Module):
    def __init__(self,
                src_vocab_size,
                embed_size,
                num_layers,
                heads,
                device,
                forward_expansion,
                dropout,
                max_length):
        super(Encoder, self).__init__()
        self.device = device
        self.word_embeddings = nn.Embedding(src_vocab_size, embed_size)
        self.positional_embeddings = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                Transformer(embed_size, heads, dropout, forward_expansion)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        out = self.dropout(self.word_embeddings(x) + self.positional_embeddings(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(DecoderBlock, self).__init__()

        self.masked_attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer = Transformer(embed_size, heads, dropout, forward_expansion)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask, trg_mask):
        masked_out = self.masked_attention(x, x, x, trg_mask)
        norm = self.dropout(masked_out + x)

        transformer_out = self.transformer(norm, encoder_out, encoder_out, src_mask)

        return transformer_out


class Decoder(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, device, num_layers, trg_vocab_size, max_length):
        super(Decoder, self).__init__()
        self.word_embeddings = nn.Embedding(trg_vocab_size, embed_size)
        self.positional_embeddings = nn.Embedding(max_length, embed_size)

        self.decoder = DecoderBlock(embed_size, heads, dropout, forward_expansion)







