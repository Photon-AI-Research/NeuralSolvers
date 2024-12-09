import torch
import numpy as np
from torch.utils.data import Dataset

class Geometry(Dataset):
    def __init__(self, lb, ub, n_points, batch_size, sampler, device):
        """
        Constructor of the Geometry class

        Args:
            lb (numpy.ndarray): lower bound of the domain.
            ub (numpy.ndarray): upper bound of the domain.
            n_points (int): the number of sampled points.
            batch_size (int): batch size
            sampler: instance of the Sampler class.
        """
        self.lb = lb
        self.ub = ub
        self.n_points = n_points
        self.batch_size = batch_size
        self.sampler = sampler
        self.device = device


    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses should implement '__getitem__' method")

    def __len__(self):
        raise NotImplementedError("Subclasses should implement '__len__' method")

