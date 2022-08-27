import torch
import numpy as np
from abc import ABC, abstractmethod

class Sampler(ABC):
    def __init__(self, n_points, batch_size):
        """
        Constructor of the Sampler class

        Args:
            n_points (int): the number of sampled points.
            batch_size (int): batch size
        """
        self.n_points = n_points
        self.batch_size = batch_size
        self.num_batches = n_points // batch_size

    @abstractmethod 
    def sample(self, lb, ub):
        """Generate sample points in [lb,ub]
        
        Args:
            lb (numpy.ndarray): lower bound of the domain.
            ub (numpy.ndarray): upper bound of the domain.
        """
        