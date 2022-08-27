import torch
import numpy as np
from torch.utils.data import Dataset

class Geometry(Dataset):
    def __init__(self, lb, ub, sampler):
        """
        Constructor of the Geometry class

        Args:
            lb (numpy.ndarray): lower bound of the domain.
            ub (numpy.ndarray): upper bound of the domain.
            sampler: instance of the Sampler class.
        """
        self.lb = lb
        self.ub = ub
        self.sampler = sampler 

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses should implement '__getitem__' method")

    def __len__(self):
        raise NotImplementedError("Subclasses should implement '__len__' method")

