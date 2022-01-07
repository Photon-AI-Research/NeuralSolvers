import torch
from torch import Tensor as Tensor
from torch.nn import Module as Module
from .LossTerm import LossTerm


class PDELoss(LossTerm):
    def __init__(self, dataset, pde, name, norm='L2', weight=1.):
        """
        Constructor of the PDE Loss

        Args:
            dataset (torch.utils.Dataset): dataset that provides the residual points
            pde (function): function that represents residual of the PDE
            norm: Norm used for calculation PDE loss
            weight: Weighting for the loss term
        """
        super(PDELoss, self).__init__(dataset, name, norm, weight)
        self.dataset = dataset
        self.pde = pde

    def __call__(self, x: Tensor, model: Module, **kwargs):
        """
        Call function of the PDE loss. Calculates the norm of the PDE residual

        x: residual points
        model: model that predicts the solution of the PDE
        """
        x.requires_grad = True  # setting requires grad to true in order to calculate
        u = model.forward(x)
        pde_residual = self.pde(x, u, **kwargs)
        zeros = torch.zeros(pde_residual.shape, device=pde_residual.device)
        return self.norm(pde_residual, zeros)


class PDELossAdaptive(PDELoss):
    def __init__(self, geometry, pde, name, sampler, norm='L2', weight=1.):
        """
        Constructor of the PDE Loss

        Args:
            dataset (torch.utils.Dataset): dataset that provides the residual points
            pde (function): function that represents residual of the PDE
            norm: Norm used for calculation PDE loss
            weight: Weighting for the loss term
        """
        super(PDELossAdaptive, self).__init__(geometry, pde, name, norm, weight)
        self.geometry = geometry
        self.sampler = sampler
        #self.pde = pde

    def __call__(self, x: Tensor, model: Module, **kwargs):
        """
        Call function of the PDE loss. Calculates the norm of the PDE residual

        x: residual points
        model: model that predicts the solution of the PDE
        """
        #x, weight = self.sampler.sample(domain=self.geometry, model=model, pde=self.pde)
        x, weight = self.sampler.sample(model=model, pde=self.pde)
        #print('x', x)
        x.requires_grad = True  # setting requires grad to true in order to calculate
        u = model.forward(x)
        pde_residual = self.pde(x, u, **kwargs)
        #zeros = torch.zeros(pde_residual.shape, device=pde_residual.device)
        #return self.norm(pde_residual, zeros)
        return 1 / self.sampler.num_points * torch.mean(1 / weight * pde_residual ** 2)