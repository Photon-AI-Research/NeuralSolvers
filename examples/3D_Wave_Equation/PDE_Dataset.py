import numpy as np
import openpmd_api as io
from torch import Tensor
from pyDOE import lhs
from torch.utils.data import Dataset


class PDEDataset(Dataset):
    def __init__(self, lb, ub, nf, batch_size, iterative_generation=False):
        """
        Constructor of the PDE Dataset

        Args:
            lb: defines the lower bound of the spatial temporal domain
            ub: defines the uppper bound of the spatial temporal domain
            nf: defines the number of residual points used in total
            batch_size: defines the number of residual points yielded in a batch
        """
        self.lb = lb
        self.ub = ub
        self.nf = nf
        self.batch_size = batch_size
        self.iterative_generation = iterative_generation

        # creating the first sampling strategy which is lhs sampling
        if not self.iterative_generation:
            self.xf = lb + (ub - lb) * lhs(4, self.nf)  # creating xf

    def __len__(self):
        """
        Returns the number of batches returned by the dataset
        """
        return self.nf // self.batch_size

    def __getitem__(self, item):
        """
        Yields the batches of xf
        """
        if not self.iterative_generation:
            return Tensor(self.xf[item * self.batch_size: (item + 1) * self.batch_size]).float()
        else:
            xf = self.lb + (self.ub - self.lb) * lhs(4, self.batch_size)  # creating xf
            return xf
