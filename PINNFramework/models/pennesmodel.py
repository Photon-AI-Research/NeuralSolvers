from torch import tanh, tensor, exp, sin, ones
from torch.nn.functional import relu
from torch.autograd import Variable
from torch.nn import Module, Parameter
from numpy import pi as Pi

class PennesHPM(Module):

    def __init__(self, heat_source_net: Module, device = 'cuda'): #k_value: Variable, p_value: Variable, heat_source_net: Module):
        """
        Constructor of the multi Model HPM
        """
        super().__init__()
        self.c_1 = 0.12 #mm2/s
        self.c_2 = 1.
        self.c_3 = 0.003 #C/s
        self.u_blood_0 = 37. #C
        self.heart_rate_value = 0.25 #Hz
        self.respiration_rate_value = 0.1 #6667 #Hz 
        self.u_ambient = 21. #C
        
        self.heat_source_net = heat_source_net
                
        #0.1, 0.01, 0, 1, 1, 0, 0 ,0.01
        self.a_1 = Parameter(Variable((0.10*ones([640,480]).clone().detach().requires_grad_(True).cuda())))
        self.a_2 = Parameter(Variable((0.00*ones([640,480]).clone().detach().requires_grad_(True).cuda())))
        self.a_3 = Parameter(Variable((0.00*ones([640,480]).clone().detach().requires_grad_(True).cuda())))
        self.a_4 = Parameter(Variable((1.00*ones([640,480]).clone().detach().requires_grad_(True).cuda())))  
        self.a_5 = Parameter(Variable((1.00*ones([640,480]).clone().detach().requires_grad_(True).cuda())))  
        self.a_6 = Parameter(Variable((0.00*ones([640,480]).clone().detach().requires_grad_(True).cuda())))
        self.a_7 = Parameter(Variable((0.00*ones([640,480]).clone().detach().requires_grad_(True).cuda())))
        self.a_8 = Parameter(Variable((0.01*ones([640,480]).clone().detach().requires_grad_(True).cuda())))
        self.a_9 = Parameter(Variable((0.10*ones([640,480]).clone().detach().requires_grad_(True).cuda())))
                   
    def convection(self, derivatives):        
        u_xx = derivatives[:, 6].view(-1)
        u_yy = derivatives[:, 7].view(-1) 
        x_indices = derivatives[:, 3].view(-1).long()
        y_indices = derivatives[:, 4].view(-1).long()
        a_5 = self.a_5[x_indices,y_indices].view(-1)
        a_5 = relu(a_5)
        return self.c_1 * a_5 * (u_xx + u_yy) 
    
    def perfusion(self, derivatives):        
        u_values = derivatives[:, 5].view(-1)    
        x_indices = derivatives[:, 3].view(-1).long()
        y_indices = derivatives[:, 4].view(-1).long()
        a_1 = self.a_1[x_indices,y_indices].view(-1)
        a_1 = relu(a_1)
        return self.c_2 * a_1 * (self.u_blood_0 - u_values)
    
    def metabolism(self, derivatives):       
        u_values = derivatives[:, 5].view(-1)
        x_indices = derivatives[:, 3].view(-1).long()
        y_indices = derivatives[:, 4].view(-1).long()
        a_4 = self.a_4[x_indices,y_indices].view(-1)
        a_4 = relu(a_4)
        return self.c_3 * a_4 * exp((u_values-self.u_blood_0)/10)
    
    def heat_source(self, derivatives):        
        heat_source_input = derivatives[:, :3]  # only positional input
        heat_source_ouput = self.heat_source_net(heat_source_input)
        heat_source_ouput = heat_source_ouput.view(-1)       
        return heat_source_ouput
       
    def respiration_rate(self, derivatives):
        t_values = derivatives[:, 2].view(-1)
        x_indices = derivatives[:, 3].view(-1).long()
        y_indices = derivatives[:, 4].view(-1).long()
        a_2 = self.a_2[x_indices,y_indices].view(-1)
        a_3 = self.a_3[x_indices,y_indices].view(-1)
        return a_2*sin(2*Pi*self.respiration_rate_value*t_values + a_3) 

    def heart_rate(self, derivatives):
        t_values = derivatives[:, 2].view(-1)
        x_indices = derivatives[:, 3].view(-1).long()
        y_indices = derivatives[:, 4].view(-1).long()
        a_6 = self.a_6[x_indices,y_indices].view(-1)
        a_7 = self.a_7[x_indices,y_indices].view(-1)
        a_8 = self.a_8[x_indices,y_indices].view(-1)
        return a_6*sin(2*Pi*self.heart_rate_value*t_values + a_7)
    
    def cooling(self, derivatives):
        x_indices = derivatives[:, 3].view(-1).long()
        y_indices = derivatives[:, 4].view(-1).long()
        u_values = derivatives[:, 5].view(-1) 
        a_9 = self.a_9[x_indices,y_indices].view(-1)
        a_9 = relu(a_9)
        return self.c_2 * a_9 * (self.u_ambient - u_values)

    def forward(self, derivatives):
        """
         Derivatives can be interpreted as complete hpm input thus it can also contain u, x, ...

         derivatives = [x,y,t,x_indices,y_indices,u,u_xx,u_yy,u_t]
        """
                 
        predicted_u_t = self.convection(derivatives) + \
                        self.perfusion(derivatives) + \
                        self.respiration_rate(derivatives) + \
                        self.heat_source(derivatives) + \
                        self.metabolism(derivatives) + \
                        self.heart_rate(derivatives) + \
                        self.cooling(derivatives)
 
        return predicted_u_t 
