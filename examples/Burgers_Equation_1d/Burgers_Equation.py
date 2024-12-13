import numpy as np
import scipy.io
import torch
from torch import Tensor, ones
from torch.autograd import grad
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from torch.profiler import profile, record_function, ProfilerActivity

import NeuralSolvers as nsolv

# Constants
DEVICE = 'cuda'
NUM_EPOCHS = 1000  # 50000
DOMAIN_LOWER_BOUND = np.array([-1, 0.0])
DOMAIN_UPPER_BOUND = np.array([1.0, 1.0])
VISCOSITY = 0.01 / np.pi
NOISE = 0.0
NUM_INITIAL_POINTS = 100
NUM_COLLOCATION_POINTS = 10000


# define underlying PDE
def burger1D(x, u):
    grads = ones(u.shape, device=u.device)  # move to the same device as prediction
    grad_u = grad(u, x, create_graph=True, grad_outputs=grads)[0]
    # calculate first order derivatives
    u_x = grad_u[:, 0]
    u_t = grad_u[:, 1]

    grads = ones(u_x.shape, device=u.device)  # move to the same device as prediction
    # calculate second order derivatives
    grad_u_x = grad(u_x, x, create_graph=True, grad_outputs=grads)[0]
    u_xx = grad_u_x[:, 0]

    # reshape for correct behavior of the optimizer
    u_x = u_x.reshape(-1, 1)
    u_t = u_t.reshape(-1, 1)
    u_xx = u_xx.reshape(-1, 1)

    f = u_t + u * u_x - (0.01 / np.pi) * u_xx
    return f


class InitialConditionDataset(Dataset):

    def __init__(self, n0, device = 'cpu',file_path = 'burgers_shock.mat'):
        """
        Constructor of the boundary condition dataset

        Args:
          n0 (int)
        """
        super(type(self)).__init__()
        self.device = device
        data = scipy.io.loadmat(file_path)

        t = data['t'].flatten()[:, None]
        x = data['x'].flatten()[:, None]

        Exact = np.real(data['usol']).T

        X, T = np.meshgrid(x, t)
        xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
        uu1 = Exact[0:1, :].T
        xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
        uu2 = Exact[:, 0:1]
        xx3 = np.hstack((X[:, -1:], T[:, -1:]))
        uu3 = Exact[:, -1:]

        X_u_train = np.vstack([xx1, xx2, xx3])
        u_train = np.vstack([uu1, uu2, uu3])

        idx = np.random.choice(X_u_train.shape[0], n0, replace=False)
        self.X_u_train = Tensor(X_u_train[idx, :]).to(self.device)
        self.u_train = Tensor(u_train[idx, :]).to(self.device)

    def __len__(self):
        """
        There exists no batch processing. So the size is 1
        """
        return 1

    def __getitem__(self, idx):
        x = self.X_u_train
        y = self.u_train

        return Tensor(x).float(), Tensor(y).float()


def load_burger_data(file_path: str = 'burgers_shock.mat'):
    """Load and return the Burgers equation data."""
    data = scipy.io.loadmat(file_path)
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    exact_solution = np.real(data['usol']).T
    return t, x, exact_solution


def setup_pinn(file_path: str = 'burgers_shock.mat'):
    """Set up and return a Physics Informed Neural Network (PINN) for solving 1D Burgers equation.

     Creates a PINN with:
     1. Initial condition dataset with NUM_INITIAL_POINTS training points
     2. Latin Hypercube Sampling (LHS) for collocation points in the domain
     3. PDE loss function for 1D Burgers equation
     4. Multi-layer perceptron (MLP) as the neural network architecture

     Architecture:
         - Input size: 2 (x, t coordinates)
         - Output size: 1 (u velocity)
         - Hidden layers: 8 layers with 40 neurons each
         - Activation: tanh
         - Domain bounds: [DOMAIN_LOWER_BOUND, DOMAIN_UPPER_BOUND]

     Components:
         - Initial Condition: Sampled from exact solution at t=0
         - Collocation Points: NUM_COLLOCATION_POINTS × NUM_COLLOCATION_POINTS grid
         - PDE Loss: Enforces Burgers equation physics
         - Boundary Conditions: None (assuming periodic or infinite domain)

     Returns:
         nsolv.PINN: Configured PINN model ready for training

     Notes:
         - Uses Latin Hypercube Sampling for optimal domain coverage
         - All computations performed on specified DEVICE (CPU/GPU)
         - Burgers equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
     """

    ic_dataset = InitialConditionDataset(n0=NUM_INITIAL_POINTS, device=DEVICE, file_path=file_path)
    initial_condition = nsolv.pinn.datasets.InitialCondition(ic_dataset, name='Initial Condition loss')

    sampler = nsolv.samplers.LHSSampler()
    geometry = nsolv.NDCube(DOMAIN_LOWER_BOUND, DOMAIN_UPPER_BOUND, NUM_COLLOCATION_POINTS, NUM_COLLOCATION_POINTS,
                            sampler, device=DEVICE)

    pde_loss = nsolv.pinn.PDELoss(geometry, burger1D, name='PDE loss')

    model = nsolv.models.mlp.MLP(
        input_size=2, output_size=1, device=DEVICE,
        hidden_size=40, num_hidden=8, lb=DOMAIN_LOWER_BOUND, ub=DOMAIN_UPPER_BOUND,
        activation=torch.tanh
    )

    return nsolv.PINN(model, 2, 1, pde_loss, initial_condition, [], device=DEVICE)


def train_pinn_profiler(pinn, num_epochs):
    """Train the PINN model."""
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_training"):
            pinn.fit(num_epochs, checkpoint_path='checkpoint.pt', restart=True,
                     logger=None, lbfgs_finetuning=False, writing_cycle=1000)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))


def train_pinn(pinn, num_epochs, logger = None):
    """Train the PINN model."""
    pinn.fit(num_epochs, checkpoint_path='checkpoint.pt', restart=True,
        logger=logger, lbfgs_finetuning=False, writing_cycle=1000)


def plot_solution(pinn, t, x, exact_solution):
    """Plot the PINN solution."""
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    pred = pinn(Tensor(X_star).to(DEVICE))
    pred = pred.detach().cpu().numpy().reshape(X.shape)

    plt.imshow(pred.T, interpolation='nearest', cmap='rainbow',
               extent=[t.min(), t.max(), x.min(), x.max()],
               origin='lower', aspect='auto')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.title(r"$u(x,t)$")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    pinn = setup_pinn()
    train_pinn(pinn, NUM_EPOCHS)

    #pinn.load_model('best_model_pinn.pt')
    t, x, exact_solution = load_burger_data()
    plot_solution(pinn, t, x, exact_solution)