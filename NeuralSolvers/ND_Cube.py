import torch
from .Geometry import Geometry

class NDCube(Geometry):
    def __init__(self, lb, ub, n_points, batch_size, sampler, device = 'cpu'):
        """
        Constructor of the NDCube class

        Args:
            lb (numpy.ndarray): lower bound of the domain.
            ub (numpy.ndarray): upper bound of the domain.
            n_points (int): the number of sampled points.
            batch_size (int): batch size
            sampler: instance of the Sampler class.
        """
        super(NDCube, self).__init__(lb, ub, n_points, batch_size, sampler, device)

        
    def __getitem__(self, idx):
        """
        Returns data at given index
        Args:
            idx (int)
        """
        self.x = self.sampler.sample(self.lb,self.ub, self.batch_size).to(self.device)
        
        if type(self.x) is tuple:
            x, w = self.x
            return torch.cat((x, w), 1)
        else:
            return self.x

    def __len__(self):
        """Length of the dataset"""
        return self.n_points // self.batch_size