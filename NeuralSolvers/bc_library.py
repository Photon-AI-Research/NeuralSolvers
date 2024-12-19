import torch

def dirichlet(x):
    return torch.zeros_like(x)[:, 0].reshape(-1, 1)