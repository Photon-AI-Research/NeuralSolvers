from NeuralSolvers.LossTerm import LossTerm
from torch import Tensor
from torch.nn import Module


class InitialCondition(LossTerm):
    def __init__(self, dataset, name, norm='L2', weight=1.):
        """
        Constructor for the Initial condition

        Args:
            dataset (torch.utils.Dataset): dataset that provides the residual points
            norm: Norm used for calculation PDE loss
            weight: Weighting for the loss term
        """
        super(InitialCondition, self).__init__(dataset, name, norm, weight)

    def __call__(self, x: Tensor, model: Module, gt_y: Tensor):
        """
        This function returns the loss for the initial condition
        L_0 = norm(model(x), gt_y)

        Args:
        x (Tensor) : position of initial condition
        model (Module): model that represents the solution
        gt_y (Tensor): ground true values for the initial state
        """
        prediction = model(x)
        return self.norm(prediction, gt_y)
