import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import RMSNorm, precompute_freqs, apply_rope

class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        # num_heads * head_dim = embedding_dim because expansion without nonlinearity is redundant and to keep compute constant across depth
        self.head_dim = embedding_dim // num_heads 
        self.num_heads = num_heads

        # no bias to save bandwidth and enable RoPE implementation
        self.qkv_proj = nn.Linear(embedding_dim, 3*embedding_dim, bias=False) # saves 2 reads of x
        self.o_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, freqs):
        batch, token, dim = x.size()

        # decompose qkv projection
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, token, 3, self.num_heads, self.head_dim) # (batch, token, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(2)

        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # (batch, num_heads, token, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # reconstruct for output projection
        x = x.transpose(1, 2).contiguous().view(batch, token, dim)

        return self.o_proj(x)

class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_ratio):
        super().__init__()

        # no bias to save bandwidth and because it has been shown to reduce training stability
        self.expand = nn.Linear(embedding_dim, embedding_dim*mlp_ratio, bias=False)
        self.compress = nn.Linear(embedding_dim*mlp_ratio, embedding_dim, bias=False)

    def forward(self, x):
        x = F.gelu(self.expand(x))
        return self.compress(x)
    
class Block(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_ratio):
        super().__init__()

        self.norm_1 = RMSNorm(embedding_dim)
        self.attention = CausalSelfAttention(embedding_dim, num_heads)

        self.norm_2 = RMSNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_ratio)

    def forward(self, x, freqs):
        x = x + self.attention(self.norm_1(x), freqs)
        return x + self.mlp(self.norm_2(x))
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, mlp_ratio, num_layers, max_seq_length):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim) # optimized version of nn.Linear assuming one hot input

        self.blocks = nn.ModuleList([Block(embedding_dim, num_heads, mlp_ratio) for _ in range(num_layers)])

        self.norm = RMSNorm(embedding_dim) # final normalization before head

        # shared weights between token embeddings and output projection
        self.output_projection = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight

        self.register_buffer("freqs", precompute_freqs(embedding_dim // num_heads, max_seq_length))

    def forward(self, index):
        t = index.size(1)
        x = self.token_embedding(index)

        freqs = self.freqs[:t] # slicing allows seq_len < max_seq_len

        for block in self.blocks:
            x = block(x, freqs)

        return self.output_projection(self.norm(x))