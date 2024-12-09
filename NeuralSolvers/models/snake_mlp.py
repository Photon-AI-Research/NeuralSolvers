import torch
import torch.nn as nn
import math
from .mlp import MLP
from .activations.snake import Snake

class SnakeMLP(MLP):
    def __init__(self, input_size, output_size, hidden_size, num_hidden, lb, ub, frequency, normalize=True):
        super(MLP, self).__init__()
        self.linear_layers = nn.ModuleList()
        self.activation = nn.ModuleList()
        self.init_layers(input_size, output_size, hidden_size, num_hidden, frequency)
        self.lb = torch.tensor(lb).float()
        self.ub = torch.tensor(ub).float()
        self.normalize = normalize
        

    def init_layers(self, input_size, output_size, hidden_size, num_hidden, frequency):
        self.linear_layers.append(nn.Linear(input_size, hidden_size))
        self.activation.append(Snake(frequency=frequency))
        for _ in range(num_hidden):
            self.linear_layers.append(nn.Linear(hidden_size, hidden_size))
            self.activation.append(Snake(frequency=frequency))
        self.linear_layers.append(nn.Linear(hidden_size, output_size))

        for m in self.linear_layers:
            if isinstance(m, nn.Linear):
                bound = math.sqrt(3 / m.weight.size()[0])
                torch.nn.init.uniform_(m.weight, a=-bound, b=bound)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.normalize:
            x = 2.0*(x - self.lb)/(self.ub - self.lb) - 1.0
        for i in range(len(self.linear_layers) - 1):
            x = self.linear_layers[i](x)
            x = self.activation[i](x)
        x = self.linear_layers[-1](x)
        return x

    def cuda(self):
        super(SnakeMLP, self).cuda()
        self.lb = self.lb.cuda()
        self.ub = self.ub.cuda()

    def cpu(self):
        super(SnakeMLP, self).cpu()
        self.lb = self.lb.cpu()
        self.ub = self.ub.cpu()

    def to(self, device):
        super(SnakeMLP, self).to(device)
        self.lb = self.lb.to(device)
        self.ub = self.ub.to(device)
