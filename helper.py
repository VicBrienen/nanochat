import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps # numerical stability constant
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # lets torch.compile fuse the entire operation into single CUDA kernel
        return output * self.weight


def precompute_freqs(dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim)) # init frequencies
    freqs = torch.outer(torch.arange(seq_len, device=freqs.device), freqs) # map to intervals
    return torch.polar(torch.ones_like(freqs), freqs) # convert to polar (enables single kernel complex fusion)

def apply_rope(x, freqs):
    B, T, H, D = x.shape
    x_complex = torch.view_as_complex(x.float().view(B, T, H, -1, 2))
    x_out = x_complex * freqs.view(1, T, 1, -1) # rotate latents
    return torch.view_as_real(x_out).view(B, T, H, D).type_as(x)