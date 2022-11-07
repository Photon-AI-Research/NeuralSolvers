from .PDELoss import PDELoss

class HPMLoss(PDELoss):
    def __init__(self, geometry, name, hpm_input, hpm_model, norm='L2', weight=1.):
        """
        Constructor of the HPM loss
        
        Args:
            geometry: instance of the geometry class that defines the domain
            hpm_input(function): function that calculates the needed input for the HPM model. The hpm_input function
            should return a list of tensors, where the last entry is the time_derivative
            hpm_model (torch.nn.Module): model for the HPM, represents the underlying PDE
            norm: Norm used for calculation PDE loss
            weight: Weighting for the loss term
        """
        super(HPMLoss, self).__init__(geometry, None, name, norm='L2', weight=1.)
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
        time_derivative = hpm_input[:, -1].reshape(-1,1)
        input = hpm_input[:, :-1]
        hpm_output = self.hpm_model(input)
        return self.norm(time_derivative, hpm_output)