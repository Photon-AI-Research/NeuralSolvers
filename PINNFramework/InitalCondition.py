from LossTerm import LossTerm
from torch import Tensor
from torch.nn import Module

class InitialCondition(LossTerm):
    def __init__(self, norm='L2'):
        super(InitialCondition, self).__init__(norm)

    def __call__(self, x: Tensor, model: Module, gt_y: Tensor):
        r"""
        This function returns the loss for the initial condition
        L_0 = norm(model(x), gt_y)

        x (Tensor) : position of initial condition
        model
        """
        prediction = model(x)
        return self.norm(prediction, gt_y)
