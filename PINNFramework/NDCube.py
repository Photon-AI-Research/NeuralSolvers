import numpy as np
import torch
from .Sampler import sample
from .Geometry import Geometry

class NDCube(Geometry):
    def __init__(self, lb, ub, num_points, sampler, num_seed = 1000, device = torch.device("cuda")):
        """
        Constructor of the NDCube class

        Args:
            lb (numpy.ndarray): lower bound of the domain.
            ub (numpy.ndarray): upper bound of the domain.
            num_points (int): the number of sampled points.
            sampler (string): sampling method: "pseudo" (pseudorandom), "LHS" (Latin hypercube sampling), and "adaptive" method.
            num_seed (int): the number of seed points for adaptive sampling
            device (torch.device): "cuda" or "cpu".
        """
            
        self.num_seed = num_seed 
        super(NDCube, self).__init__(lb, ub, num_points, sampler, device)

    def sample_points(self, model, pde):
        print('self.sampler',self.sampler)
        print('self.device',self.device)
        print('type device',type(self.device))
        return sample(self.lb, self.ub, model, pde, self.num_points, self.sampler, self.num_seed, self.device)
