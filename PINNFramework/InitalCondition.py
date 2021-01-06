from LossTerm import LossTerm
from torch import Tensor
from torch.nn import Module


class InitialCondition(LossTerm):
    def __init__(self, dataset, norm='L2', weight=1.):
        super(InitialCondition, self).__init__(dataset, norm, weight)

    def __call__(self, x: Tensor, model: Module, gt_y: Tensor):
        r"""
        This function returns the loss for the initial condition
        L_0 = norm(model(x), gt_y)

        x (Tensor) : position of initial condition
        model
        """
        prediction = model(x)
        return self.weight * self.norm(prediction, gt_y)
