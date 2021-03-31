from .PDELoss import PDELoss
from torch.autograd import grad
from torch import ones

class HPMLoss(PDELoss):
    def __init__(self, dataset, hpm_input, hpm_model, norm='L2', weight=1., weight_j=0.01):
        """
        Constructor of the HPM loss

        Args:
            dataset (torch.utils.Dataset): dataset that provides the residual points
            hpm_input(function): function that calculates the needed input for the HPM model. The hpm_input function
            should return a list of tensors, where the last entry is the time_derivative
            hpm_model (torch.nn.Module): model for the HPM, represents the underlying PDE
            norm: Norm used for calculation PDE loss
            weight: Weighting for the loss term
            weight_j: Weighting for the heat source regularization term
        """
        super(HPMLoss, self).__init__(dataset, None, norm, weight)
        self.hpm_input = hpm_input
        self.hpm_model = hpm_model
        self.weight_j = weight_j

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
        time_derivative = hpm_input[:, -1]
        input = hpm_input[:, :-1]
        hpm_output = self.hpm_model(input)
        
        # heat source regularizetion term
        x = x[:,:3]
        heat_source_output = self.hpm_model.heat_source_net(x)
        grads = ones(heat_source_output.shape, device=heat_source_output.device)
        du_dx_values = grad(heat_source_output, x, create_graph=True, grad_outputs=grads)[0]
        u_x_values = du_dx_values[:, 0].view(-1)
        u_y_values = du_dx_values[:, 1].view(-1)
                
        return self.weight * self.norm(time_derivative, hpm_output) + self.weight_j * (u_x_values.abs() + u_y_values.abs()).sum()
