import sys

from argparse import ArgumentParser
import numpy
import numpy as np
import scipy.io
from pyDOE import lhs
import torch
from torch import Tensor, ones, stack, load
from torch.autograd import grad
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import wandb

sys.path.append("NeuralSolvers/")  # PINNFramework etc.
import PINNFramework as pf


class BoundaryConditionDatasetlb(Dataset):

    def __init__(self, nb, lb, ub):
        """
        Constructor of the lower boundary condition dataset

        Args:
          nb (int)
          lb (numpy.ndarray)
          ub (numpy.ndarray)
        """
        super(type(self)).__init__()

        # maximum of the time domain
        max_t = 2
        t = np.linspace(0, max_t, 200).flatten()[:, None]
        idx_t = np.random.choice(t.shape[0], nb, replace=False)
        tb = t[idx_t, :]
        self.x_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)

    def __getitem__(self, idx):
        """
        Returns data at given index
        Args:
            idx (int)
        """
        return Tensor(self.x_lb).float()

    def __len__(self):
        """
        There exists no batch processing. So the size is 1
        """
        return 1


class BoundaryConditionDatasetub(Dataset):

    def __init__(self, nb, lb, ub):
        """
        Constructor of the upper boundary condition dataset

        Args:
          nb (int)
          lb (numpy.ndarray)
          ub (numpy.ndarray)
        """
        super(type(self)).__init__()

        # maximum of the time domain
        max_t = 2
        t = np.linspace(0, max_t, 200).flatten()[:, None]
        idx_t = np.random.choice(t.shape[0], nb, replace=False)
        tb = t[idx_t, :]
        self.x_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)

    def __getitem__(self, idx):
        """
        Returns data at given index
        Args:
            idx (int)
        """
        return Tensor(self.x_ub).float()

    def __len__(self):
        """
        There exists no batch processing. So the size is 1
        """
        return 1


class InitialConditionDataset(Dataset):

    def __init__(self, n0):
        """
        Constructor of the inital condition dataset

        Args:
          n0 (int)
        """
        super(type(self)).__init__()

        L = 1
        c = 1
        alpha = (c * np.pi / L) ** 2
        max_t = 10
        max_x = L

        t = np.zeros(200).flatten()[:, None]
        x = np.linspace(0, max_x, 200).flatten()[:, None]

        U = (np.exp(-(alpha) * t)) * np.sin(np.pi * x / L)
        u = U.flatten()[:, None]

        idx_x = np.random.choice(x.shape[0], n0, replace=False)
        self.x = x[idx_x, :]
        self.u = u[idx_x, :]
        self.t = t[idx_x, :]

    def __len__(self):
        """
        There exists no batch processing. So the size is 1
        """
        return 1

    def __getitem__(self, idx):
        x = np.concatenate([self.x, self.t], axis=1)
        y = np.concatenate([self.u], axis=1)
        return Tensor(x).float(), Tensor(y).float()


class PDEDataset(Dataset):

    def __init__(self, nf, lb, ub):
        """
        Constructor of the PDE dataset

        Args:
          nf (int)
          lb (numpy.ndarray)
          ub (numpy.ndarray)
        """
        self.xf = lb + (ub - lb) * lhs(2, nf)

    def __getitem__(self, idx):
        """
        Returns data at given index
        Args:
            idx (int)
        """
        return Tensor(self.xf).float()

    def __len__(self):
        """
        There exists no batch processing. So the size is 1
        """
        return 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=10000,
                        help='Number of training iterations')
    parser.add_argument('--n0', dest='n0', type=int, default=50, help='Number of input points for initial condition')
    parser.add_argument('--nb', dest='nb', type=int, default=50, help='Number of input points for boundary condition')
    parser.add_argument('--nf', dest='nf', type=int, default=20000, help='Number of input points for pde loss')
    parser.add_argument('--num_hidden', dest='num_hidden', type=int, default=4, help='Number of hidden layers')
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=100, help='Size of hidden layers')
    parser.add_argument('--annealing', dest='annealing', type=int, default=0, help='Enables annealing with 1')
    parser.add_argument('--annealing_cycle', dest='annealing_cycle', type=int, default=5, help='Cycle of lr annealing')
    parser.add_argument('--track_gradient', dest='track_gradient', default=1, help='Enables tracking of the gradients')
    args = parser.parse_args()
    # Domain bounds
    lb = np.array([0, 0.0])
    ub = np.array([1.0, 2.0])

    # initial condition
    ic_dataset = InitialConditionDataset(n0=args.n0)
    initial_condition = pf.InitialCondition(ic_dataset, name='Initial condition')

    # boundary conditions
    bc_datasetlb = BoundaryConditionDatasetlb(nb=args.nb, lb=lb, ub=ub)
    bc_datasetub = BoundaryConditionDatasetub(nb=args.nb, lb=lb, ub=ub)


    # Function for dirichlet boundary condition
    def func(x):
        return torch.zeros_like(x)[:, 0].reshape(-1, 1)


    dirichlet_bc_u_lb = pf.DirichletBC(func, bc_datasetlb, name='ulb dirichlet boundary condition')
    dirichlet_bc_u_ub = pf.DirichletBC(func, bc_datasetub, name='uub dirichlet boundary condition')

    # PDE
    pde_dataset = PDEDataset(args.nf, lb, ub)


    def heat1d(x, u):
        grads = ones(u.shape, device=u.device)  # move to the same device as prediction
        grad_u = grad(u, x, create_graph=True, grad_outputs=grads)[0]

        # calculate first order derivatives
        u_x = grad_u[:, 0]
        u_t = grad_u[:, 1]

        # calculate second order derivatives
        grads = ones(u_x.shape, device=u.device)  # move to the same device as prediction
        grad_u_x = grad(u_x, x, create_graph=True, grad_outputs=grads)[0]
        u_xx = grad_u_x[:, 0]

        # reshape for correct behavior of the optimizer
        u_x = u_x.reshape(-1, 1)
        u_t = u_t.reshape(-1, 1)
        u_xx = u_xx.reshape(-1, 1)

        # residual function
        f = u_t - 1 * u_xx

        return f


    pde_loss = pf.PDELoss(pde_dataset, heat1d, name='1D Heat', weight=1)

    # create model
    model = pf.models.MLP(input_size=2,
                          output_size=1,
                          hidden_size=args.hidden_size,
                          num_hidden=args.num_hidden,
                          lb=lb,
                          ub=ub)

    # create PINN instance
    pinn = pf.PINN(model, 2, 1, pde_loss, initial_condition, [dirichlet_bc_u_lb, dirichlet_bc_u_ub], use_gpu=True)

    # load the pretrained model
    pinn.load_model('best_model_pinn.pt')  # <- use "1D_Heat_Equation.py" script to generate this file

    # Plotting
    max_t = 2
    max_x = 1

    t = np.linspace(0, max_t, 200).flatten()[:, None]
    x = np.linspace(0, max_x, 200).flatten()[:, None]
    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    # Setup for the distributed inference
    infer_ = pf.DistributedInfer(pinn.model,
                                 save_inferences=True,
                                 dir_name="heat_eq_output",
                                 use_horovod=False
                                 )
    # pred = pinn(Tensor(X_star).cuda())    <- for normal inference
    pred = infer_.multi_infer(Tensor(X_star).cuda())  # <- for distributed inference across multiple nodes
    pred_all = infer_.combine_numpy_files()

    H_pred = pred_all.reshape(X.shape)
    plt.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu',
               extent=[lb[1], ub[1], lb[0], ub[0]],
               origin='lower', aspect='auto')
    plt.ylabel('x (cm)')
    plt.xlabel('t (seconds)')
    plt.colorbar().set_label('Temperature (°C)')
    plt.show()

