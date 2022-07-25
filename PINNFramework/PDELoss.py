import torch
from torch import Tensor as Tensor
from torch.nn import Module as Module
from .LossTerm import LossTerm


class PDELoss(LossTerm):
    def __init__(self, geometry, pde, name, norm='L2', weight=1.):
        """
        Constructor of the PDE Loss

        Args:
            geometry: instance of the geometry class that defines the domain
            pde (function): function that represents residual of the PDE
            norm: Norm used for calculation PDE loss
            weight: Weighting for the loss term
        """
        super(PDELoss, self).__init__(geometry, name, norm, weight)
        self.geometry = geometry
        self.pde = pde

    def __call__(self, x: Tensor, model: Module, **kwargs):
        """
        Call function of the PDE loss. Calculates the norm of the PDE residual

        x: residual points
        model: model that predicts the solution of the PDE
        """

        if self.geometry.sampler == 'adaptive':
            x,w = x
        
        x.requires_grad = True  # setting requires grad to true in order to calculate
        u = model.forward(x)
        pde_residual = self.pde(x, u, **kwargs)
        
        if self.geometry.sampler == 'adaptive':
            return 1 / self.geometry.n_points * torch.mean(1 / w * pde_residual ** 2)
        else:
            zeros = torch.zeros(pde_residual.shape, device=pde_residual.device)
            return self.norm(pde_residual, zeros)