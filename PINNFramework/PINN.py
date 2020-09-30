import torch
import torch.nn as nn


class Interface(nn.Module):
    
    def __init__(self, model, input_d, output_d):
        super(Interface, self).__init__()
        self.model = model 
        self.input_d = input_d
        self.output_d = output_d
        self.hpm_model = None
    
    def forward(self,x):
        """ Forward step of the PINN """
        x = self.input_normalization(x)
        return self.model(x)

    def pinn_loss(self,
                  x, ex_u, boundary_u,
                  interpolation_criterion, boundary_criterion, pde_norm, 
                  lambda_0=1., lambda_b=1., lambda_f=1.):
        """ Calculating PINN loss"""
        x_0 = x["x_0"]
        x_b = x["x_b"]
        x_f = x["x_f"]
        len_x0 = x_0.shape[0]
        len_xb = x_b.shape[0]
        len_xf = x_f.shape[0]
        input_x = torch.cat([x_0, x_b, x_f]).float()
        input_x.requires_grad = True
        prediction_u = self.forward(input_x)
        u_0 = prediction_u[:len_x0,:]
        u_b = prediction_u[len_x0:-len_xf,:]
        u_f = prediction_u[-len_xf:]
        pred_derivatives = self.derivatives(prediction_u, input_x)
        l_0 = self.interpolation_loss(x_0, u_0,interpolation_criterion)
        l_b = self.boundary_loss(x_b, u_b, boundary_criterion)
        l_f = self.pde_loss(input_x, prediction_u, pred_derivatives, pde_norm)
        return lambda_0 * l_0 + lambda_b * l_b * lambda_f 
        

    def boundary_loss(self, u, boundary_u, criterion):
        """ Calculation of the boundary loss"""
        return criterion(boundary_u, u)
    
    def interpolation_loss(self, pred_u, exact_u, criterion):
        """ Calculation of the boundary loss"""
        return criterion(exact_u, pred_u)

    def pde_loss(self, x, u, derivatives, norm):
        """ Calculation of the pde loss TODO: move the torch zeros to the same device"""
        if self.hpm_model:
            return norm(
                derivatives[:,-self.output_d:]-self.hpm_model(x,u,derivatives),
                torch.zeros([x.shape[0],self.output_d]))
        else:
            return norm(
                derivatives[:,-self.output_d:] - self.pde(x,u,derivatives),
                torch.zeros([x.shape[0],self.output_d]))
        
    def pde(self, x, u, derivatives):
        """
        Formulation of the right hand side of the pde
        """
        raise NotImplementedError

    def derivatives(self,u, x):
        """ Calculates necessary derivatives """
        raise NotImplementedError
    
    def set_hpm(self, model):
        """
        Setting the HPM model to the model
        """
        self.hpm_model = model
    
    def input_normalization(self,x):
        """
        Implementation of the input_normalization
        """
        raise NotImplementedError
    
    def unset_hpm(self):
        """
        Unset HPM functionality fo the model
        """
        self.hpm_model = None 
    
    
    

    
