import numpy as np
import torch
from .Sampler import sample
from .Geometry import Geometry

class NDCube(Geometry):
    def __init__(self, lb, ub, num_points, sampler, num_seed = 1000, device = torch.device("cuda")):
        self.num_seed = num_seed 
        super(NDCube, self).__init__(lb, ub, num_points, sampler, device)

    def sample_points(self, model, pde):
        print('self.sampler',self.sampler)
        print('self.device',self.device)
        print('type device',type(self.device))
        return sample(self.lb, self.ub, model, pde, self.num_points, self.sampler, self.num_seed, self.device)
