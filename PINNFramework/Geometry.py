import torch
import numpy as np
from abc import ABC, abstractmethod

class Geometry(ABC):
    def __init__(self, lb, ub, n_points, sampler= 'LHS', device = torch.device("cuda")):
        """
        Constructor of the Geometry class

        Args:
            lb (numpy.ndarray): lower bound of the domain.
            ub (numpy.ndarray): upper bound of the domain.
            n_points (int): the number of sampled points.
            sampler (string): sampling method: "random" (pseudorandom), "LHS" (Latin hypercube sampling), and "adaptive" method.
            device (torch.device): "cuda" or "cpu".
        """
        self.lb = lb
        self.ub = ub
        self.n_points = n_points
        self.sampler = sampler
        self.device = device

    @abstractmethod 
    def sample_points(self):
        """Sample points within the geometry."""