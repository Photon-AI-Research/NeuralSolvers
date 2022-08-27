import torch
import numpy as np
from .Sampler import Sampler


class RandomSampler(Sampler):
    def __init__(self, n_points, batch_size):
        """
        Constructor of the RandomSampler (pseudo random sampler) class

        Args:
            n_points (int): the number of sampled points.
            batch_size (int): batch size
        """
        super(RandomSampler, self).__init__(n_points, batch_size)
        

    def sample(self, lb, ub):
        """Generate sample points in [lb,ub]
        
        Args:
            lb (numpy.ndarray): lower bound of the domain.
            ub (numpy.ndarray): upper bound of the domain.
        """
            
        torch.manual_seed(42)
        np.random.seed(42)
        
        lb =  lb.reshape(1,-1)
        ub =  ub.reshape(1,-1)
        
        dimension = lb.shape[1]
        xf = np.random.uniform(lb,ub,size=(self.n_points, dimension))
        return torch.tensor(xf).float()