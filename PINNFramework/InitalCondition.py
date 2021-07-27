from .LossTerm import LossTerm
from torch.nn import L1Loss
from torch import Tensor, ones_like, zeros_like
from torch.nn import Module
from torch.autograd import grad


class InitialCondition(LossTerm):
    def __init__(self, dataset, name, norm='L2', weight=1., weight_j=1.):
        """
        Constructor for the Initial condition
        Args:
            dataset (torch.utils.Dataset): dataset that provides the residual points
            norm: Norm used for calculation PDE loss
            weight: Weighting for the loss term
            weight_j: Weighting for the du/dt regularizer loss term
        """
        super(InitialCondition, self).__init__(dataset, name, norm, weight)
        self.norm_j = L1Loss()
        self.weight_j = weight_j

    def __call__(self, x: Tensor, model: Module, gt_y: Tensor):
        """
        This function returns the loss for the initial condition
        L_0 = norm(model(x), gt_y)
        L_j = L1(u_tt, 0)
        Args:
        x (Tensor) : position of initial condition
        model (Module): model that represents the solution
        gt_y (Tensor): ground true values for the initial state
        """
        # if weight_j != 0 calculate d^2u/dt^2 regularizer loss term
        if (self.weight_j > 0.):
            x.requires_grad = True
            prediction = model(x)
            grads = ones_like(prediction)
            du_dx_values = grad(
                prediction,
                x,
                create_graph=True,
                grad_outputs=grads)[0]
            u_t_values = du_dx_values[:, 2].reshape(prediction.shape)

            u_tt_values = grad(
                u_t_values, x, create_graph=True, grad_outputs=grads)[0]
            u_tt_values = u_tt_values[:, 2].reshape(prediction.shape)
            return self.weight*self.norm(prediction, gt_y) + self.weight_j*self.norm_j(u_tt_values, zeros_like(u_tt_values))
        else: 
            prediction = model(x)
            return self.weight*self.norm(prediction, gt_y)