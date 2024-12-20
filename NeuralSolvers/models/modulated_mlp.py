import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, vit_l_16, vit_l_32
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from .mlp import set_seed

class ModulatedMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden, lb, ub,
                 activation=torch.tanh, normalize=True, nfeat_mod = 128, device='cpu'):
        """
        MLP with bias modulation using embeddings from a ViT model.

        Args:
            input_size (int): Input dimensionality (e.g., x, t).
            output_size (int): Output dimensionality (e.g., u(x, t)).
            hidden_size (int): Hidden layer size.
            num_hidden (int): Number of hidden layers.
            lb (list): Lower bounds for normalization.
            ub (list): Upper bounds for normalization.
            u_i (Tensor): reference solution (expected to be 2d)
            activation (callable): Activation function for hidden layers.
            normalize (bool): Whether to normalize inputs.
            device (str): Device to use ('cpu' or 'cuda').
        """
        set_seed(2342)
        super(ModulatedMLP, self).__init__()
        self.linear_layers = nn.ModuleList()
        self.activation = activation
        self.lb = torch.Tensor(lb).float()
        self.ub = torch.Tensor(ub).float()
        self.normalize = normalize
        self.device = device
        self.nfeat_mod = nfeat_mod
        # Modulation network to process spatio-temporal information of h_1
        self.modulation_network = nn.Sequential(
            nn.Linear(hidden_size, self.nfeat_mod),
            nn.ReLU(),
            nn.Linear(self.nfeat_mod, hidden_size * num_hidden)
        )

        self.init_layers(input_size, output_size, hidden_size, num_hidden)

    def init_layers(self, input_size, output_size, hidden_size, num_hidden):
        self.linear_layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_hidden):
            self.linear_layers.append(nn.Linear(hidden_size, hidden_size))
        self.linear_layers.append(nn.Linear(hidden_size, output_size))

        for m in self.linear_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        for m in self.modulation_network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        """
        Forward pass with ViT-based bias modulation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output of the MLP with modulated biases.
        """

        noPoints, _ = x.shape

        # Check device alignment
        if x.device != self.device:
            warnings.warn(f"Input tensor was on {x.device}, but model is on {self.device}. "
                          f"Input tensor has been moved to {self.device}. "
                          "This may slow down computation. Consider moving your input tensor to the correct device before calling the model.",
                          UserWarning)
            x = x.to(self.device)

        # Normalize inputs
        if self.normalize:
            x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0

        # Compute activation of first layer
        x = self.linear_layers[0](x)
        x = self.activation(x)

        # Compute modulation terms
        modulation_terms = self.modulation_network(x)  # Shape: (batch_size, hidden_size * num_hidden)
        modulation_terms = torch.sigmoid(modulation_terms)

        modulation_terms = modulation_terms.view(-1, len(self.linear_layers) - 2, self.linear_layers[1].bias.size(0))

        '''
        # Forward pass with bias modulation
        # Start with i = 1
        for i in range(len(self.linear_layers) - 2):
            x = self.linear_layers[i+1](x)
            bias = self.linear_layers[i+1].bias + modulation_terms[:, i]
            x = self.activation(x + bias)  # Apply modulated bias
        '''

        # Forward pass with full weight modulation
        for i in range(len(self.linear_layers) - 2):
            wi = self.linear_layers[i + 1].weight  # Shape: [out_features, in_features]
            bi = self.linear_layers[i + 1].bias  # Shape: [out_features]
            modi = modulation_terms[:, i]  # Shape: [batch_size, out_features]

            x = x * modi

            # Standard linear transformation
            x = x + torch.matmul(x, wi.T) + bi  # Shape: [batch_size, out_features]

            # Activation
            x = self.activation(x)

        # Final layer
        x = self.linear_layers[-1](x)

        return x


    def to(self, device):
        super(ModulatedMLP, self).to(device)
        self.lb = self.lb.to(device)
        self.ub = self.ub.to(device)