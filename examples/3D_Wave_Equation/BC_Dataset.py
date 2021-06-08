import numpy as np
import openpmd_api as io
from torch import Tensor
from pyDOE import lhs
from torch.utils.data import Dataset
from torch import randint


class BoundaryDataset(Dataset):
    def __init__(self, lb, ub, nb, batch_size, period=1):
        """
        Constructor of the PDE Dataset

        Args:
            lb: defines the lower bound of the spatial temporal domain
            ub: defines the uppper bound of the spatial temporal domain
            nb: defines the number of residual points used in total
            batch_size: defines the number of residual points yielded in a batch
            period: defines how many periods the spatial domain extended gets extended
        """
        self.lb = lb
        self.ub = ub
        self.nb = nb
        self.batch_size = batch_size
        domain_size = ub - lb
        # creating the first sampling strategy which is lhs sampling
        self.x_lb = Tensor(lb + (ub - lb) * lhs(4, self.nb)).float()  # creating nb sampling points
        r = randint(1, period+1, (self.nb, 4)).float()
        r[:, -1] = 0 # no time shift
        self.x_ub = self.x_lb + r * domain_size

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
