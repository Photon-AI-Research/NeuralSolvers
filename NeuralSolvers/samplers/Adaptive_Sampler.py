import torch
import numpy as np
from NeuralSolvers.samplers.Sampler import Sampler


class AdaptiveSampler(Sampler):
    def __init__(self, n_seed, model, pde, device = torch.device("cuda")):
        """
        Constructor of the AdaptiveSampler class

        Args:            
            n_seed (int): the number of seed points for adaptive sampling.
            model: is the model which is trained to represent the underlying PDE.
            pde (function): function that represents residual of the PDE.
            device (torch.device): "cuda" or "cpu".
        """
        self.n_seed = n_seed
        self.model = model 
        self.pde = pde 
        self.device = device 
        super(AdaptiveSampler, self).__init__()

    def sample(self, lb, ub, n):
        """
        Generate a tuple of 'n' sampled points in [lb,ub] and corresponding weights.
        
        Args:
            lb (numpy.ndarray): lower bound of the domain.
            ub (numpy.ndarray): upper bound of the domain.
            n (int): the number of sampled points.
        """
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        lb =  lb.reshape(1,-1)
        ub =  ub.reshape(1,-1)

        dimension = lb.shape[1]
        xs = np.random.uniform(lb, ub, size=(self.n_seed, dimension))

        # collocation points
        xf = np.random.uniform(lb, ub, size=(n, dimension))

        # make the points into tensors
        xf = torch.tensor(xf).float().to(self.device)
        xs = torch.tensor(xs).float().to(self.device)

        # prediction with seed points
        xs.requires_grad = True
        prediction_seed = self.model(xs)

        # pde residual with seed points
        loss_seed = self.pde(xs, prediction_seed)
        losses_xf = torch.zeros_like(xf)

        # Compute the 2-norm distance between seed points and collocation points
        dist = torch.cdist(xf, xs, p=2)

        # obtain the smallest element of the given tensor
        knn = dist.topk(1, largest=False)

        # assign the seed loss to the loss of the closest collocation points
        losses_xf = loss_seed[knn.indices[:, 0]]

        # apply softmax function
        q_model = torch.softmax(losses_xf, dim=0)

        # obtain 'n' indices sampled from the multinomial distribution
        indicies_new = torch.multinomial(q_model[:, 0], n, replacement=True)

        # collocation points and corresponding weights
        xf = xf[indicies_new]
        weight = q_model[indicies_new].detach()
        weight = torch.mean(weight, 1, True)

        return xf, weight