import torch
import numpy as np

class InitialConditionDataset(torch.utils.data.Dataset):
    """Generalized Initial Condition Dataset."""

    def __init__(self, n0, initial_func, domain, device='cpu'):
        x = np.linspace(domain[0][0], domain[1][0], n0)[:, None]
        u0 = initial_func(x)
        self.X_u_train = torch.Tensor(np.hstack((x, np.zeros_like(x)))).to(device)
        self.u_train = torch.Tensor(u0).to(device)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.X_u_train.float(), self.u_train.float()