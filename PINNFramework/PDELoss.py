import torch 

class PDELoss():
    def pde(self, x, u, derivatives):
        raise NotImplementedError("Definition of the PDE is not implemented")

    
    def derivatives(self,x, u):
        raise NotImplementedError ("Calculation of the derivatives has to be defined")
    
    def __call__(self, x, model_u):
        x.requires_grad = True # seeting requires grad to true in order to calculate
        u = model_u.forward(x)
        derivatives = self.derivatives(x,u)
        return self.pde(x,u,derivatives)

