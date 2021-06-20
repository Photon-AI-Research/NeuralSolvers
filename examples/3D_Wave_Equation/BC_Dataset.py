import numpy as np
import openpmd_api as io
from torch import Tensor
from pyDOE import lhs
from torch.utils.data import Dataset
from torch import randint


class BoundaryDataset(Dataset):
    def __init__(self, lb, ub, nb, batch_size, direction=1):
        """
        Constructor of the PDE Dataset

        Args:
            lb: defines the lower bound of the spatial temporal domain
            ub: defines the uppper bound of the spatial temporal domain
            nb: defines the number of residual points used in total
            batch_size: defines the number of residual points yielded in a batch
            period: defines how many periods the spatial domain extended gets extended
        """
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.nb = nb
        self.batch_size = batch_size
        domain_size = self.ub - self.lb
        # creating the first sampling strategy which is lhs sampling
        self.x_lb = Tensor(self.lb + (self.ub - self.lb) * lhs(4, self.nb)).float()  # creating nb sampling points
        self.x_lb[:, direction] = self.lb[direction]
        mask = np.zeros(self.lb.shape)
        mask[direction] = 1
        self.x_ub = self.x_lb + mask * self.ub

    def __len__(self):
        """
        Returns the number of batches returned by the dataset
        """
        return self.nb // self.batch_size

    def __getitem__(self, item):
        """
        Yields the batches of xf
        """
        x_lb = self.x_lb[item * self.batch_size: (item + 1) * self.batch_size]
        x_ub = self.x_ub[item * self.batch_size: (item + 1) * self.batch_size]

        return x_lb, x_ub
