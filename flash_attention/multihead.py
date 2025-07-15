from torch import nn
import torch
import torch.nn.functional as F
from time import perf_counter

device = torch.device('cpu')

embedding_size = 1024
block_size = 2048
batch_size = 32
dropout = 0.5

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embedding_size, head_size) # (C, C)
        self.query = nn.Linear(embedding_size, head_size) # (C, C)
        self.value = nn.Linear(embedding_size, head_size) # (C, C)

        # create a tril mask using register buffer, such that the tensor can be moved to GPU, and is not learnable
        self.register_buffer("tril", torch.ones(block_size, block_size))

        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        K = self.key(x) # Key matrix (B, T, C)
        Q = self.query(x) # Query matrix (B, T, C)
        wei = Q @ K.transpose(1, 2) * self.head_size ** -0.5 # attention matrix (B, T, T)
        wei = wei.masked_fill(self.tril==0, float('-inf'))
        F.softmax(wei, dim=-1) # (B, T, T)
        # wei = self.dropout(wei)
        V = self.value(x) # (B, T, C)
        return wei @ V # (B, T, C)


class MultiheadAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        assert embedding_size % num_heads == 0
        # each K, Q, V has shape (C, C / num_heads)
        self.heads = nn.ModuleList([Head(embedding_size // num_heads) for _ in range(num_heads)])
        # redundant matrix, (C, C)
        self.proj = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # cat num_heads (B, T, C / num_heads) together to get (B, T, C)
        out = self.dropout(self.proj(out))
        return out



x = torch.randn((batch_size, block_size, embedding_size))
head = MultiheadAttention(num_heads=8)
print(x.shape)

start = perf_counter()
head.forward(x)
end = perf_counter()

print(f'took {(end-start)*1000:.3f}ms')






