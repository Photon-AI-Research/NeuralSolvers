from torch.utils.data import Dataset
import numpy as np
from torch import Tensor, ones, stack, load

class Geometry(Dataset):
    def __init__(self, lb, ub):

        #self.xf = lb + (ub - lb) * lhs(2, nf)
        self.xf = np.concatenate([lb.reshape(1,-1), ub.reshape(1,-1)], axis = 0)

    def __getitem__(self, idx):

        return Tensor(self.xf).float()

    def __len__(self):

        return 1