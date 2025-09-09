import torch

def causal_mask(seq_len, device):
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
