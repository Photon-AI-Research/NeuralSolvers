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
        self.lb = lb
        self.ub = ub
        self.xf = np.concatenate([lb.reshape(1,-1), ub.reshape(1,-1)], axis = 0)

    def __getitem__(self, idx):

        return Tensor(self.xf).float()

    def __len__(self):

        return 1