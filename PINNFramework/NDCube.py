import numpy as np
import torch
from .Sampler import sample
from .Geometry import Geometry

class NDCube(Geometry):
    def __init__(self, lb, ub, n_points, sampler, n_seed = 1000, device = torch.device("cuda")):
        """
        Constructor of the NDCube class

        Args:
            lb (numpy.ndarray): lower bound of the domain.
            ub (numpy.ndarray): upper bound of the domain.
            n_points (int): the number of sampled points.
            sampler (string): "random" (pseudorandom), "LHS" (Latin hypercube sampling),
            and "adaptive" sampling methods.
            n_seed (int): the number of seed points for adaptive sampling
            device (torch.device): "cuda" or "cpu".
        """
            
        self.n_seed = n_seed 
        super(NDCube, self).__init__(lb, ub, n_points, sampler, device)

    def sample_points(self, model, pde):
        return sample(self.lb, self.ub, self.n_points, self.sampler, self.n_seed, model, pde, self.device)
