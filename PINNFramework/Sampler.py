import numpy as np
import torch
from torch import Tensor, ones, stack, load
from pyDOE import lhs

# set initial seed for torch and numpy
seed = 42


def sample(lb, ub, n_points, sampler, n_seed, model, pde, device = torch.device("cuda:0")):
    """Generate sample points in [lb,ub] using random, LHS or adaptive sampling methods. Returns either sampled points ("random", "LHS") or concatenated tensor of sampled points with corresponding weights ("adaptive").
    
    Args:
        lb (numpy.ndarray): lower bound of the domain.
        ub (numpy.ndarray): upper bound of the domain.
        n_points (int): the number of sampled points.
        sampler (string): sampling method: "random" (pseudorandom), "LHS" (Latin hypercube sampling), and "adaptive" method.
        n_seed (int): the number of seed points for adaptive sampling.
        model: is the model which is trained to represent the underlying PDE.
        pde (function): function that represents residual of the PDE.
        device (torch.device): "cuda" or "cpu".
    """
    lb =  lb.reshape(1,-1)
    ub =  ub.reshape(1,-1)

    if sampler == "random":
        return pseudorandom(lb, ub, n_points)
    if sampler == "LHS":
        return quasirandom(lb, ub, n_points)
    if sampler =='adaptive':
        return adaptive(lb, ub, n_points, n_seed, model, pde, device)
    raise ValueError(f"{sampler} sampler is not available.")
    
    
def pseudorandom(lb, ub, n_points):
    """Pseudo random sampling. Returns sampled points in [lb,ub].
    Args:
        lb (numpy.ndarray): lower bound of the domain.
        ub (numpy.ndarray): upper bound of the domain.
        n_points (int): the number of sampled points.
    """
    
    np.random.seed(seed)
    dimension = lb.shape[1]
    xf = np.random.uniform(lb,ub,size=(n_points, dimension))
    return torch.tensor(xf).float()


def quasirandom(lb, ub, n_points):
    """LHS sampling. Returns sampled points in [lb,ub].
    Args:
        lb (numpy.ndarray): lower bound of the domain.
        ub (numpy.ndarray): upper bound of the domain.
        n_points (int): the number of sampled points.
    """
    np.random.seed(seed)
    dimension = lb.shape[1]
    xf = lb + (ub - lb) * lhs(dimension, n_points)
    return torch.tensor(xf).float()
    
def adaptive(lb, ub, n_points, n_seed, model, pde, device):
    """Adaptive sampling. Returns concatenated tensor of sampled points in [lb,ub] and corresponding weights.
    Args:
        lb (numpy.ndarray): lower bound of the domain.
        ub (numpy.ndarray): upper bound of the domain.
        n_points (int): the number of sampled points.
        n_seed (int): the number of seed points for adaptive sampling.
        model: is the model which is trained to represent the underlying PDE.
        pde (function): function that represents residual of the PDE.
        device (torch.device): "cuda" or "cpu".
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    dimension = lb.shape[1]
    xs = np.random.uniform(lb, ub, size=(n_seed, dimension))

    # collocation points
    xf = np.random.uniform(lb, ub, size=(n_points, dimension))

    # make the points into tensors
    xf = torch.tensor(xf).float().to(device)
    xs = torch.tensor(xs).float().to(device)

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

    # obtain 'n_points' indices sampled from the multinomial distribution
    indicies_new = torch.multinomial(q_model[:, 0], n_points, replacement=True)

    # collocation points and corresponding weights
    xf = xf[indicies_new]
    weight = q_model[indicies_new].detach()
    return torch.cat((xf,weight), 1)