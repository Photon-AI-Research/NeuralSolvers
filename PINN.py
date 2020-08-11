import torch.nn as nn

class Interface(nn.Module):

    def forward(self):
        """ Forward step of the PINN """
        raise NotImplementedError

    def pinn_loss(self):
        """ Calculating pinn loss"""
        raise NotImplementedError
    
    def boundary_loss(self):
        """ Calculation of the boundary loss"""
        raise NotImplementedError
    
    def initial_loss(self):
        """ Calculation of the boundary loss"""
        raise NotImplementedError

    def pde_loss(self):
        """ Calculation of the pde loss """
        raise NotImplementedError

    def derivatives(self):
        """ Calculates necessary derivatives """
        raise NotImplementedError
    
    

    
