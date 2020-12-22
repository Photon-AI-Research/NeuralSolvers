import torch
from torch import Tensor as Tensor
from torch.nn import Module as Module
from torch.nn import MSELoss, L1Loss
from LossTerm import LossTerm


class PDELoss(LossTerm):
    def __init__(self, norm='L2'):
        # cases for standard torch norms
        if norm == 'L2':
            self.norm = MSELoss()
        elif norm == 'L1':
            self.norm = L1Loss()
        else:
            # Case for self implemented norms TODO: add documentation or a guide
            self.norm = norm

    def pde(self, x: Tensor, u: Tensor, derivatives: Tensor):
        """
        Implements the underlying PDE f(x) = 0

        x: residual points
        u: prediction of the model
        derivatives: partial derivatives
        """
        raise NotImplementedError("Definition of the PDE is not implemented")

    def derivatives(self, x: Tensor, u: Tensor):
        raise NotImplementedError("Calculation of the derivatives has to be defined")

    def __call__(self, x: Tensor, model: Module):
        """
        Call function of the PDE loss. Calculates the norm of the PDE residual

        x: residual points
        model: model that predicts the solution of the PDE
        """
        x.requires_grad = True  # setting requires grad to true in order to calculate
        u = model.forward(x)
        derivatives = self.derivatives(x, u)
        pde_residual = self.pde(x, u, derivatives)
        zeros = torch.zeros(pde_residual.shape, device=pde_residual.device)
        return self.norm(pde_residual, zeros)
