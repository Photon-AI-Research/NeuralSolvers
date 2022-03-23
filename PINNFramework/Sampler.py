import numpy as np
import torch
from torch import Tensor, ones, stack, load
from pyDOE import lhs
import matplotlib.pyplot as plt

# set initial seed for torch and numpy
torch.manual_seed(42)
np.random.seed(42)

class Sampler:
    def __init__(self, geometry, num_points, ns, sampler="adaptive", device = 'cuda'):
        """
        Constructor of the Sampler

        Args:
            geometry (torch.utils.Dataset): provides the geometry of the domain
            num_points (int): number of collocation points
            ns (int): number of seed points
            sampler (str): type of sampling
            device: gpu or cpu
        """
        self.geometry = geometry
        self.num_points = num_points
        self.ns = ns
        self.sampler = sampler
        self.device = device
   
    def sample(self, model, pde):
        
      if self.sampler == "adaptive":
        lb = self.geometry.xf[0]
        ub = self.geometry.xf[1]

        # random seeds
        random_x = np.random.uniform(lb[0], ub[0], self.ns).reshape(-1, 1)
        random_t = np.random.uniform(lb[1], ub[1], self.ns).reshape(-1, 1)

        xs = np.concatenate([random_x,random_t], axis=1)

        # collocation points    
        random_x = np.random.uniform(lb[0], ub[0], self.num_points).reshape(-1, 1)
        random_t = np.random.uniform(lb[1], ub[1], self.num_points).reshape(-1, 1)
        xf = np.concatenate([random_x, random_t], axis=1)

        # make the points into tensors
        xf = torch.tensor(xf).float().to(self.device)
        xs = torch.tensor(xs).float().to(self.device)
        
        # prediction with seed points
        xs.requires_grad = True
        prediction_seed = model(xs)
        
        # pde residual with seed points
        loss_seed = pde(xs, prediction_seed)
        losses_xf = torch.zeros_like(xf)
        
        # Compute the 2-norm distance between seed points and collocation points
        dist = torch.cdist(xf, xs, p=2)

        # obtain the smallest element of the given tensor
        knn = dist.topk(1, largest=False)
        
        # assign the seed loss to the loss of the closest collocation points
        losses_xf = loss_seed[knn.indices[:, 0]]

        # apply softmax function
        q_model = torch.softmax(losses_xf, dim=0)

        # obtain 'num_points' indices sampled from the multinomial distribution
        indicies_new = torch.multinomial(q_model[:, 0], self.num_points, replacement=True)
        
        # collocation points and corresponding weights
        self.xf = xf[indicies_new]
        self.weight = q_model[indicies_new]

      return self.xf, self.weight


