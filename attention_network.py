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

        value = self.values(value)
        key = self.keys(key)
        query = self.queries(query)
        energy = torch.einsum("nqhd, nkhd->nhqk", [query, key])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        #output is softmax(q*kT/ (k**(1/2)) )V
        output = torch.einsum("nhql,nlhd->nqhd", [attention, value]).reshape(N, query_len, self.heads*self.head_dim)
        out = self.fc_out(output)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
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
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
            for _ in range(num_layers)]
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
        '''
        O(t-1) -> Masked Self attention --> Self Attention
        '''
        super(DecoderBlock, self).__init__()
        self.masked_attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.TransformerBlock = TransformerBlock(embed_size, heads, dropout, forward_expansion)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key, value, src_mask, trg_mask):
        masked_out = self.masked_attention(x, x, x, trg_mask)
        norm = self.dropout(masked_out + x)

        TransformerBlock_out = self.TransformerBlock(norm, key, value, src_mask)

        return TransformerBlock_out


class Decoder(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, device, num_layers, trg_vocab_size, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embeddings = nn.Embedding(trg_vocab_size, embed_size)
        self.positional_embeddings = nn.Embedding(max_length, embed_size)

        self.decoder_layers = nn.ModuleList([DecoderBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.softmax = nn.Softmax()
    def forward(self, x, encoder_out, src_mask, trg_mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        out = self.dropout(self.word_embeddings(x) + self.positional_embeddings(positions))

        for layer in self.decoder_layers:
            out = layer(out, encoder_out, encoder_out, src_mask, trg_mask)

        out = self.fc_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, src_pad_idx, trg_pad_idx, trg_vocab_size, embed_size=256, heads=8, dropout=0, forward_expansion=4, device="cuda", num_layers=6, max_length=100):
        super(Transformer, self).__init__()
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, 
                               heads, device, forward_expansion,
                               dropout, max_length)
        
        self.decoder = Decoder(embed_size, heads, dropout, forward_expansion,
                               device, num_layers, trg_vocab_size, max_length)

    def set_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def set_trg_mask(self, trg):
        trg_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.set_src_mask(src)
        trg_mask = self.set_trg_mask(trg)
        out = self.encoder(src, src_mask)
        print("Decoder Output")
        print(out.shape)
        out = self.decoder(trg, out, src_mask, trg_mask)

        return out









