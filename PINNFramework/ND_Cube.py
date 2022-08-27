import numpy as np
import torch
from .Geometry import Geometry

class NDCube(Geometry):
    def __init__(self, lb, ub, sampler):
        """
        Constructor of the NDCube class

        Args:
            lb (numpy.ndarray): lower bound of the domain.
            ub (numpy.ndarray): upper bound of the domain.
            sampler: instance of the Sampler class.
        """
        super(NDCube, self).__init__(lb,ub,sampler)
        
        
    def __getitem__(self, idx):
        """
        Returns data at given index
        Args:
            idx (int)
        """
        self.x = self.sampler.sample(self.lb,self.ub) 
        
        if type(self.x) is tuple:
            x, w = self.x
            x = x[idx * self.sampler.batch_size: (idx + 1) * self.sampler.batch_size]
            w = w[idx * self.sampler.batch_size: (idx + 1) * self.sampler.batch_size]
            return torch.cat((x, w), 1)
        else:           
            return self.x[idx * self.sampler.batch_size: (idx + 1) * self.sampler.batch_size]

    def __len__(self):
        """Length of the dataset"""
        return self.sampler.num_batches