from torch.utils.data import Dataset
import numpy as np
from torch import Tensor, ones, stack, load

class Geometry(Dataset):
    def __init__(self, lb, ub):
        """
        Constructor of the Geometry class

        Args:
            lb (numpy.ndarray): lower bound of the domain
            ub (numpy.ndarray): upper bound of the domain
        """
        self.lb = lb.reshape(1,-1)
        self.ub = ub.reshape(1,-1)
        self.xf = np.concatenate([self.lb, self.ub], axis = 0)

    def __getitem__(self, idx):

        return Tensor(self.xf).float()

    def __len__(self):

        return 1