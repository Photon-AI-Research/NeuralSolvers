import numpy as np
import torch
from torch import Tensor, ones, stack, load
from pyDOE import lhs

class Sampler:
    def __init__(self, geometry, num_points, sampler="lhs"):
        self.geometry = geometry
        self.num_points = num_points
        self.sampler = sampler
   
    def sample(self, model, pde):

      if self.sampler == "lhs":
        np.random.seed(42)
        lb = self.geometry.xf[0]
        ub = self.geometry.xf[1]

        xf = lb + (ub - lb) * lhs(2, self.num_points)
        weight = np.ones_like(xf, shape =self.num_points) 

        self.xf= torch.tensor(xf).float()  
        self.weight = torch.tensor(weight).float()
        
      if self.sampler == "adaptive":
        np.random.seed(42)
        ns = int(self.num_points/8)
        lb = self.geometry.xf[0]
        ub = self.geometry.xf[1]

        # random seeds
        random_x = np.random.uniform(lb[0], ub[0], ns).reshape(-1, 1)
        random_t = np.random.uniform(lb[1], ub[1], ns).reshape(-1, 1)

        #random_x = np.linspace(lb[0], ub[0], int(np.sqrt(ns)))
        #random_t = np.linspace(lb[1], ub[1], int(np.sqrt(ns)))

        #x_grid, t_grid = np.meshgrid(random_x, random_t)
        #xs = np.concatenate([x_grid.reshape(-1,1), t_grid.reshape(-1, 1)], axis=1)
        xs = np.concatenate([random_x,random_t], axis=1)

        # collocation points    
        random_x = np.random.uniform(lb[0], ub[0], self.num_points).reshape(-1, 1)
        random_t = np.random.uniform(lb[1], ub[1], self.num_points).reshape(-1, 1)
        xf = np.concatenate([random_x, random_t], axis=1)

        # make them into tensors
        xf = torch.tensor(xf).float().cuda()
        xs = torch.tensor(xs).float().cuda()
        
        xs.requires_grad = True
        # predictions seed
        prediction_seed = model(xs)
        print('xs', xs.shape)
        
        loss_seed = pde(xs, prediction_seed)
        losses_xf = torch.zeros_like(xf)
        dist = torch.cdist(xf, xs, p=2)
        knn = dist.topk(1, largest=False)
        losses_xf = loss_seed[knn.indices[:, 0]]
        q_model = torch.softmax(losses_xf, dim=0)
        indicies_new = torch.multinomial(q_model[:, 0], self.num_points, replacement=True)

        self.xf = xf[indicies_new]
        self.weight = q_model[indicies_new]

      return self.xf.cuda(), self.weight.cuda()


