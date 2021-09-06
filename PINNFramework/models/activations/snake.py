import torch
import torch.nn as nn


class Snake(nn.Module,): 
    """ Implementation of the snake activation function as a torch nn module 
    The result of the activation function a(x) is calculated by a(x) = x + sin^2(x)
    With alpha is a trainab
    """

    def __init__(self,frequency=10): 
        """Constructor function that initialize the torch module
        """
        super(Snake, self).__init__() 
    
        # making beta trainable by activating gradient calculation
        self.a = nn.Parameter(torch.tensor([float(frequency)], requires_grad=True))
        
    def forward(self, x): 
        return  x + ((torch.sin(self.a* x)) ** 2) / self.a