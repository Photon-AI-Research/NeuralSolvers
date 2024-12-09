import warnings

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden, lb, ub, activation=torch.tanh, normalize=True, device='cpu'):
        super(MLP, self).__init__()
        self.linear_layers = nn.ModuleList()
        self.activation = activation
        self.init_layers(input_size, output_size, hidden_size,num_hidden)
        self.lb = torch.Tensor(lb).float()
        self.ub = torch.Tensor(ub).float()
        self.normalize = normalize
        self.device = device

    def init_layers(self, input_size, output_size, hidden_size, num_hidden):
        self.linear_layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_hidden):
            self.linear_layers.append(nn.Linear(hidden_size, hidden_size))
        self.linear_layers.append(nn.Linear(hidden_size, output_size))

        for m in self.linear_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.device != self.device:
            warnings.warn(f"Input tensor was on {x.device}, but model is on {self.device}. "
                          f"Input tensor has been moved to {self.device}. "
                          "This may slow down computation. Consider moving your input tensor to the correct device before calling the model.",
                          UserWarning)
            x = x.to(self.device)

        if self.normalize:
            x = 2.0*(x - self.lb)/(self.ub - self.lb) - 1.0
        for i in range(len(self.linear_layers) - 1):
            x = self.linear_layers[i](x)
            x = self.activation(x)
        x = self.linear_layers[-1](x)
        return x

    def cuda(self):
        super(MLP, self).cuda()
        self.lb = self.lb.cuda()
        self.ub = self.ub.cuda()

    def cpu(self):
        super(MLP, self).cpu()
        self.lb = self.lb.cpu()
        self.ub = self.ub.cpu()
        
    def to(self, device):
        super(MLP, self).to(device)
        self.lb = self.lb.to(device)
        self.ub = self.ub.to(device)
