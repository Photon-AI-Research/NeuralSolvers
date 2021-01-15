from .PDELoss import PDELoss

class HPMLoss(PDELoss):
    def __init__(self, dataset, derivatives, hpm_model, norm='L2', weight=1.):
        """
        Constructor of the HPM loss

        Args:
            dataset (torch.utils.Dataset): dataset that provides the residual points
            derivatives (function): function that calculates the needed PDES for the HPM model. The derivatives function
            should return a list of tensors, where the last entry is the time_derivative
            hpm_model (torch.nn.Module): model for the HPM, represents the underlying PDE
            norm: Norm used for calculation PDE loss
            weight: Weighting for the loss term
        """
        super(HPMLoss, self).__init__(dataset, None, norm, weight)
        self.derivatives = derivatives
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
        derivatives = self.derivatives(x, prediction_u)
        hpm_output = self.hpm_model(derivatives)
        self.norm(derivatives[:, -1], hpm_output)
