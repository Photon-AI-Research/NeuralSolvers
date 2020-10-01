from PINNFramework.PINN import Interface


import torch
from torch.autograd import grad


class KS_PINN(Interface):

    def __init__(self, model, input_d, output_d):
        """
        Invoke the super contructior of the interface class.
        :param model: PyTorch model
        :param input_d: input dimension of the model
        :param output_d: output dimension of the model
        """

        super(KS_PINN).__init__(model, input_d, output_d)

    def forward(self, x):
        """
        Forward step in the NN without data normalization.
        :param x:
        """
        return self.model(x)

    def derivatives(self, u, x):

        """
        Obtain the derivatives from the forward step.
        :param u: Obtained forward data points
        :param x:
        """

        gradients = torch.ones(x.shape[0]) # Not sure if this work correctly
        pred_phi = u[:, 0] # get the predicted values

        # Use autograd to get the KS Orbital gradient
        Dphi = grad(pred_phi, x, create_graph=True, grad_outputs=gradients)[0]

        Dphi_x = Dphi[:, 0]
        Dphi_y = Dphi[:, 1]
        Dphi_z = Dphi[:, 2]

        # These operations give you the x,y,z derivative components of the corresponding laplace operator.
        Lap_phi_x = (grad(Dphi_x, x, create_graph=True, grad_outputs=gradients)[0])[:, 0]
        Lap_phi_y = (grad(Dphi_y, x, create_graph=True, grad_outputs=gradients)[0])[:, 1]
        Lap_phi_z = (grad(Dphi_z, x, create_graph=True, grad_outputs=gradients)[0])[:, 2]

        derivatives = torch.stack([Dphi_x, Dphi_y, Dphi_z, Lap_phi_x, Lap_phi_y, Lap_phi_z], 1)

        return derivatives

