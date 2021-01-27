from .PDELoss import PDELoss

class HPMLoss(PDELoss):
    def __init__(self, dataset, hpm_input, hpm_model, norm='L2', weight=1.):
        """
        Constructor of the HPM loss

        Args:
            dataset (torch.utils.Dataset): dataset that provides the residual points
            hpm_input(function): function that calculates the needed input for the HPM model. The hpm_input function
            should return a list of tensors, where the last entry is the time_derivative
            hpm_model (torch.nn.Module): model for the HPM, represents the underlying PDE
            norm: Norm used for calculation PDE loss
            weight: Weighting for the loss term
        """
        super(HPMLoss, self).__init__(dataset, None, norm, weight)
        self.hpm_input = hpm_input
        self.hpm_model = hpm_model

    def __call__(self, x, model, **kwargs):
        """
        Calculation of the HPM Loss

        Args:
            x(torch.Tensor): residual points
            model(torch.nn.module): model representing the solution
        """
        x.requires_grad = True
        prediction_u = model(x)
        hpm_input = self.hpm_input(x, prediction_u)
        hpm_output = self.hpm_model(hpm_input)
        return self.weight * self.norm(hpm_input[:, -1], hpm_output)
