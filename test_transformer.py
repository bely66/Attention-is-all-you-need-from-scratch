import torch
from attention_network import Transformer
device = torch.device('cpu')

x = torch.tensor([[1,3,7,4,6,5,2,0,0], [1,3,9,7,8,4,6,5,2]])

trg = torch.tensor([[1,3,7,4,2,5,2,0,0], [1,3,9,7,8,4,6,5,2]])


src_pad_idx = 0
trg_pad_idx = 0

src_vocab_size = 10 
trg_vocab_size = 10

model = Transformer(src_vocab_size, src_pad_idx, trg_pad_idx, trg_vocab_size, device=device)

out = model(x, trg[:, :-1])
print(out)