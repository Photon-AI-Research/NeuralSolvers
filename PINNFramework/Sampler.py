import torch
import numpy as np
from abc import ABC, abstractmethod

class Sampler(ABC):
    def __init__(self):
        """
        Constructor of the Sampler class
        """

    @abstractmethod 
    def sample(self, lb, ub, n):
        """Generate 'n' number of sample points in [lb,ub]
        
        Args:
            lb (numpy.ndarray): lower bound of the domain.
            ub (numpy.ndarray): upper bound of the domain.
            n (int): the number of sampled points.
        """
        