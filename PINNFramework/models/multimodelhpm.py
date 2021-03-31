from torch import tanh
from torch.nn import Module


class MultiModelHPM(Module):

    def __init__(self, alpha_net: Module, heat_source_net: Module):
        """
        Constructor of the multi Model HPM
        """
        super(MultiModelHPM, self).__init__()
        self.alpha_net = alpha_net
        self.heat_source_net = heat_source_net

    def forward(self, derivatives):
        """
         Derivatives can be interpreted as complete hpm input thus it can also contain u, x, ...

         x = [x,y,t]
         derivatives = [x,y,t,u,u_xx,u_yy,u_t]
        """
        alpha_net_input = derivatives[:, :3]
        alpha_output = self.alpha_net(alpha_net_input)
        alpha_output = alpha_output.view(-1) #alpha_output.view(3,-1) 
        alpha_output = 0.1 + 0.05 * tanh(alpha_output) #alpha, alpha_blood, u_blood = 

        heat_source_input = derivatives[:, :2]  # only positional input
        heat_source_ouput = self.heat_source_net(heat_source_input)
        heat_source_ouput = heat_source_ouput.view(-1)

        #u = derivatives[:,3].view(-1)
        u_xx = derivatives[:, 4].view(-1)
        u_yy = derivatives[:, 5].view(-1)

        predicted_u_t = alpha_output * (u_xx + u_yy) + heat_source_ouput # + alpha_blood*(u_blood-u)

        return predicted_u_t
    
    def cuda(self):
        super(MLP, self).cuda()
        self.lb = self.lb.cuda()
        self.ub = self.ub.cuda()

    def cpu(self):
        super(MLP, self).cpu()
        self.lb = self.lb.cpu()
        self.ub = self.ub.cpu()
