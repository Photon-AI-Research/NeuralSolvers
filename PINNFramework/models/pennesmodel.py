from torch import tanh, tensor, exp
from torch.autograd import Variable
from torch.nn import Module, Parameter


class PennesHPM(Module):

    def __init__(self, heat_source_net: Module, device = 'cuda'): #k_value: Variable, p_value: Variable, heat_source_net: Module):
        """
        Constructor of the multi Model HPM
        """
        super().__init__()
        self.c_1 = 0.12 #mm2/s
        self.c_2 = 1 
        self.c_3 = 240 #mm3*C/J
        self.u_blood = 37 #C
        self.heat_source_net = heat_source_net
        
        self.a_1 = Parameter(tensor(-350., requires_grad=True, device = device))
        self.a_2 = Parameter(tensor(-11., requires_grad=True, device = device))
        self.a_3 = Parameter(tensor(0.01, requires_grad=True, device = device))
        self.a_4 = Parameter(tensor(0.1, requires_grad=True, device = device))
        
    def convection(self, derivatives):
        
        u_xx = derivatives[:, 6].view(-1)
        u_yy = derivatives[:, 7].view(-1)
        
        return self.c_1 * (u_xx + u_yy)
    
    def perfusion(self, derivatives):
        
        u_values = derivatives[:, 5].view(-1)
        
        return self.c_2 * (self.a_1 + self.a_2*u_values) * (self.u_blood - u_values)
    
    def metabolism(self, derivatives):
        
        u_values = derivatives[:, 5].view(-1)
        
        return self.c_3 * self.a_3 * exp(-self.a_4/u_values)
    
    def heat_source(self, derivatives):
        
        heat_source_input = derivatives[:, :2]  # only positional input
        heat_source_ouput = self.heat_source_net(heat_source_input)
        heat_source_ouput = heat_source_ouput.view(-1)
        
        return heat_source_ouput

    def forward(self, derivatives):
        """
         Derivatives can be interpreted as complete hpm input thus it can also contain u, x, ...

         x = [x,y,t]
         derivatives = [x,y,t,x_indices,y_indices,u,u_xx,u_yy,u_t]
        """
                 
        predicted_u_t = self.convection(derivatives) + \
                        self.perfusion(derivatives) + \
                        self.metabolism(derivatives) + \
                        self.heat_source(derivatives)

        return predicted_u_t

