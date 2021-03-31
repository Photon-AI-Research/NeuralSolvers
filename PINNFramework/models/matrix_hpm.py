from torch import tanh
from torch.autograd import Variable
from torch.nn import Module, Parameter


class MatrixHPM(Module):

    def __init__(self, k_values: Variable, heat_source_net: Module):
        """
        Constructor of the multi Model HPM
        """
        super().__init__()
        self.k_values = Parameter(k_values)
        self.heat_source_net = heat_source_net

    def forward(self, derivatives):
        """
         Derivatives can be interpreted as complete hpm input thus it can also contain u, x, ...

         x = [x,y,t]
         derivatives = [x,y,t,x_indices,y_indices,u,u_xx,u_yy,u_t]
        """
        heat_source_input = derivatives[:, :3]  # only positional input
        heat_source_ouput = self.heat_source_net(heat_source_input)
        heat_source_ouput = heat_source_ouput.view(-1)

        #u = derivatives[:,3].view(-1)
        x_indices = derivatives[:, 3].view(-1).long()
        y_indices = derivatives[:, 4].view(-1).long()
        u_xx = derivatives[:, 6].view(-1)
        u_yy = derivatives[:, 7].view(-1)
        k_values = self.k_values[x_indices,y_indices].view(-1)

        predicted_u_t = k_values * (u_xx + u_yy) + heat_source_ouput # + alpha_blood*(u_blood-u)

        return predicted_u_t
    
    def cuda(self):
        super(MLP, self).cuda()
        self.lb = self.lb.cuda()
        self.ub = self.ub.cuda()

    def cpu(self):
        super(MLP, self).cpu()
        self.lb = self.lb.cpu()
        self.ub = self.ub.cpu()
