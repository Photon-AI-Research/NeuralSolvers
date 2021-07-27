from torch import tanh, tensor, exp, sin, ones, where, zeros_like
from torch.nn.functional import relu
from torch.autograd import Variable
from torch.nn import Module, Parameter
from numpy import pi as Pi
from numpy import vectorize


class PennesHPM_(Module):

    def __init__(self, config, heat_source_net=None): 
        """
        Constructor of the multi Model HPM
        Args:
            config(dict): dictionary defining the configuration of the model
            heat_source_net(nn.Module): MLP to account for residuals
        """
        super().__init__()
        self.config = config

        if config['convection']:
            self.c_1 = 0.12 #mm2/s
            self.a_5 = Parameter(Variable((1.00*ones([640,480]).clone().detach().requires_grad_(True))))  
        if config['perfusion']:
            self.u_blood_0 = 37. #C
            self.c_2 = 1.
            self.a_1 = Parameter(Variable((0.10*ones([640,480]).clone().detach().requires_grad_(True))))
        if config['metabolism']:
            self.u_blood_0 = 37. #C
            self.c_3 = 0.003 #C/s
            self.a_4 = Parameter(Variable((1.00*ones([640,480]).clone().detach().requires_grad_(True))))  
        if config['heat_source']:
            self.heat_source_net = heat_source_net
        if config['respiration']:
            self.respiration_rate_value = 0.1 #0.16667 #Hz 
            self.a_2 = Parameter(Variable((0.00*ones([640,480]).clone().detach().requires_grad_(True))))
            self.a_3 = Parameter(Variable((0.00*ones([640,480]).clone().detach().requires_grad_(True))))
        if config['heart_rate']:
            self.heart_rate_value = 0.25 #Hz
            self.a_6 = Parameter(Variable((0.00*ones([640,480]).clone().detach().requires_grad_(True))))
            self.a_7 = Parameter(Variable((0.00*ones([640,480]).clone().detach().requires_grad_(True))))
        if config['cooling']:
            self.c_2 = 1.
            self.u_ambient = 21. #C
            self.a_9 = Parameter(Variable((0.10*ones([640,480]).clone().detach().requires_grad_(True))))       
        if config['cold_bolus']:
            self.t_delay = Parameter(Variable((0.0*ones([640,480]).clone().detach().requires_grad_(True))))
            self.t_cooling = Parameter(Variable((1.0*ones([640,480]).clone().detach().requires_grad_(True))))
            self.u_cool_factor = Parameter(Variable((0.8*ones([640,480]).clone().detach().requires_grad_(True))))
            self.alpha = Parameter(Variable((1.0*ones([640,480]).clone().detach().requires_grad_(True))))
            
    def cuda(self):
        super().cuda()
        if self.config['convection']:
            self.a_5 = self.a_5.cuda()
        if self.config['perfusion']:
            self.a_1 = self.a_1.cuda()
        if self.config['metabolism']:
            self.a_4 = self.a_4.cuda()
        if self.config['heat_source']:
            self.heat_source_net.cuda()
        if self.config['respiration']:
            self.a_2 = self.a_2.cuda()
            self.a_3 = self.a_3.cuda()
        if self.config['heart_rate']:
            self.a_6 = self.a_6.cuda()
            self.a_7 = self.a_7.cuda()
        if self.config['cooling']:
            self.a_9 = self.a_9.cuda()
        if self.config['cold_bolus']:
            self.t_delay = self.t_delay.cuda()
            self.t_cooling = self.t_cooling.cuda()
            self.u_cool_factor = self.u_cool_factor.cuda()
            self.alpha = self.alpha.cuda()

    def u_blood(self, t_values, x_indices, y_indices):
        if not self.config['cold_bolus']:
            return zeros_like(t_values) + self.u_blood_0
        else: 
            t_delay = self.t_delay[x_indices,y_indices].view(-1)
            t_cooling = self.t_cooling[x_indices,y_indices].view(-1)
            alpha = self.alpha[x_indices,y_indices].view(-1)
            u_cool = relu(self.u_cool_factor[x_indices,y_indices].view(-1))*self.u_blood_0
            return self.u_blood_t(t_values, t_delay, t_cooling, -alpha, alpha, u_cool, self.u_blood_0)

    def u_blood_t(self, t, t_delay, t_cooling, a1, a2, u_cool, u_blood_0):
        b1 = u_blood_0 - a1*t_delay
        t1 = (u_cool - b1)/a1
        t2 = t1 + t_cooling
        b2 = u_cool - a2*t2
        t3 = (u_blood_0 - b2)/a2
        ind1 = where(t < t_delay)
        ind2 = where((t >= t_delay) + (t < t1))
        ind3 = where((t >= t1) + (t < t2))
        ind4 = where((t < t3) + (t >= t2))
        ind5 = where(t >= t3)
        t[ind1] = u_blood_0
        t[ind2] = b1[ind2]+a1[ind2]*t[ind2]
        t[ind3] = u_cool
        t[ind4] = b2[ind4]+a2[ind4]*t[ind4]
        t[ind5] = u_blood_0
        return t
                   
    def convection(self, derivatives):        
        u_xx = derivatives[:, 6].view(-1)
        u_yy = derivatives[:, 7].view(-1) 
        x_indices = derivatives[:, 3].view(-1).long()
        y_indices = derivatives[:, 4].view(-1).long()
        a_5 = relu(self.a_5[x_indices,y_indices].view(-1))
        return self.c_1 * a_5 * (u_xx + u_yy) 
    
    def is_vessel(self, derivatives, threshold = -0.5):
        u_xx = derivatives[:, 6].view(-1)
        u_yy = derivatives[:, 6].view(-1)
        return where((u_xx + u_yy) < threshold)
    
    def perfusion(self, derivatives):   
        ind = self.is_vessel(derivatives)
        t_values = derivatives[:, 2].view(-1)
        x_indices = derivatives[:, 3].view(-1).long()
        y_indices = derivatives[:, 4].view(-1).long()
        u_values = derivatives[:, 5].view(-1)    
        a_1 = relu(self.a_1[x_indices,y_indices].view(-1))
        u_blood = self.u_blood(t_values, x_indices, y_indices)
        perfusion = zeros_like(u_values)
        perfusion[ind] = self.c_2 * a_1[ind] * (u_blood[ind] - u_values[ind]) 
        return perfusion
    
    def metabolism(self, derivatives):       
        u_values = derivatives[:, 5].view(-1)
        x_indices = derivatives[:, 3].view(-1).long()
        y_indices = derivatives[:, 4].view(-1).long()
        t_values = derivatives[:, 2].view(-1)
        a_4 = relu(self.a_4[x_indices,y_indices].view(-1))
        u_blood = self.u_blood(t_values, x_indices, y_indices)
        return self.c_3 * a_4 * exp((u_values-u_blood)/10)
    
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
        return a_6*sin(2*Pi*self.heart_rate_value*t_values + a_7)
    
    def cooling(self, derivatives):
        x_indices = derivatives[:, 3].view(-1).long()
        y_indices = derivatives[:, 4].view(-1).long()
        u_values = derivatives[:, 5].view(-1) 
        a_9 = relu(self.a_9[x_indices,y_indices].view(-1))
        return self.c_2 * a_9 * (self.u_ambient - u_values)

    def forward(self, derivatives):
        """
         Derivatives can be interpreted as complete hpm input thus it can also contain u, x, ...

         derivatives = [x,y,t,x_indices,y_indices,u,u_xx,u_yy,u_t]
        """               
        predicted_u_t = 0
        if self.config['convection']:
            predicted_u_t += self.convection(derivatives)
        if self.config['perfusion']:
            predicted_u_t += self.perfusion(derivatives)
        if self.config['metabolism']:
            predicted_u_t += self.metabolism(derivatives)
        if self.config['heat_source']:
            predicted_u_t += self.heat_source(derivatives)
        if self.config['respiration']:
            predicted_u_t += self.respiration_rate(derivatives)
        if self.config['heart_rate']:
            predicted_u_t += self.heart_rate(derivatives)
        if self.config['cooling']:
            predicted_u_t += self.cooling(derivatives)  
        return predicted_u_t 