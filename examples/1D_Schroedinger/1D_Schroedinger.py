import sys
from argparse import ArgumentParser

import numpy as np
import scipy.io
from pyDOE import lhs
from torch import Tensor, ones, stack, load
from torch.autograd import grad
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append('NeuralSolvers/')  # PINNFramework etc.
import PINNFramework as pf


class BoundaryConditionDataset(Dataset):

    def __init__(self, nb, lb, ub):
        """
        Constructor of the initial condition dataset

        Args:
          n0 (int)
        """
        super(type(self)).__init__()
        data = scipy.io.loadmat('NLS.mat')
        t = data['tt'].flatten()[:, None]
        idx_t = np.random.choice(t.shape[0], nb, replace=False)
        tb = t[idx_t, :]
        self.x_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
        self.x_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)

    def __getitem__(self, idx):
        """
        Returns data for initial state
        """
        return Tensor(self.x_lb).float(), Tensor(self.x_ub).float()

    def __len__(self):
        """
        There exists no batch processing. So the size is 1
        """
        return 1


class InitialConditionDataset(Dataset):

    def __init__(self, n0):
        """
        Constructor of the boundary condition dataset

        Args:
          n0 (int)
        """
        super(type(self)).__init__()
        data = scipy.io.loadmat('NLS.mat')
        x = data['x'].flatten()[:, None]
        t = data['tt'].flatten()[:, None]
        Exact = data['uu']
        Exact_u = np.real(Exact)
        Exact_v = np.imag(Exact)
        idx_x = np.random.choice(x.shape[0], n0, replace=False)
        self.x = x[idx_x, :]
        self.u = Exact_u[idx_x, 0:1]
        self.v = Exact_v[idx_x, 0:1]
        self.t = np.zeros(self.x.shape)

    def __len__(self):
        """
        There exists no batch processing. So the size is 1
        """
        return 1

    def __getitem__(self, idx):
        x = np.concatenate([self.x, self.t], axis=1)
        y = np.concatenate([self.u, self.v], axis=1)
        return Tensor(x).float(), Tensor(y).float()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=10000, help='Number of training iterations')
    parser.add_argument('--n0', dest='n0', type=int, default=50, help='Number of input points for initial condition')
    parser.add_argument('--nb', dest='nb', type=int, default=50, help='Number of input points for boundary condition')
    parser.add_argument('--nf', dest='nf', type=int, default=20000, help='Number of input points for pde loss')
    parser.add_argument('--nf_batch', dest='nf_batch', type=int, default=20000, help='Batch size for sampler')
    parser.add_argument('--num_hidden', dest='num_hidden', type=int, default=4, help='Number of hidden layers')
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=100, help='Size of hidden layers')
    parser.add_argument('--annealing', dest='annealing', type=int, default=0, help='Enables annealing with 1')
    parser.add_argument('--annealing_cycle', dest='annealing_cycle', type=int, default=5, help='Cycle of lr annealing')
    parser.add_argument('--track_gradient', dest='track_gradient', default=1, help='Enables tracking of the gradients')
    args = parser.parse_args()
    # Domain bounds
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi / 2])
    # initial condition
    ic_dataset = InitialConditionDataset(n0=args.n0)
    initial_condition = pf.InitialCondition(ic_dataset, name='Initial condition')
    # boundary conditions
    bc_dataset = BoundaryConditionDataset(nb=args.nb, lb=lb, ub=ub)
    periodic_bc_u = pf.PeriodicBC(bc_dataset, 0, "u periodic boundary condition")
    periodic_bc_v = pf.PeriodicBC(bc_dataset, 1, "v periodic boundary condition")
    periodic_bc_u_x = pf.PeriodicBC(bc_dataset, 0, "u_x periodic boundary condition", 1, 0)
    periodic_bc_v_x = pf.PeriodicBC(bc_dataset, 1, "v_x periodic boundary condition", 1, 0)
    
    #sampler
    sampler = pf.LHSSampler(n_points = args.nf, batch_size = args.nf_batch)
    #sampler = pf.RandomSampler(n_points= args.nf, batch_size = args.nf_batch)
    
    # geometry
    geometry = pf.NDCube(lb,ub,sampler)

    def schroedinger1d(x, u):
        pred = u
        u = pred[:, 0]
        v = pred[:, 1]
        
        grads = ones(u.shape, device=pred.device) # move to the same device as prediction
        grad_u = grad(u, x, create_graph=True, grad_outputs=grads)[0]
        grad_v = grad(v, x, create_graph=True, grad_outputs=grads)[0]

        # calculate first order derivatives
        u_x = grad_u[:, 0]
        u_t = grad_u[:, 1]

        v_x = grad_v[:, 0]
        v_t = grad_v[:, 1]

        # calculate second order derivatives
        grad_u_x = grad(u_x, x, create_graph=True, grad_outputs=grads)[0]
        grad_v_x = grad(v_x, x, create_graph=True, grad_outputs=grads)[0]

        u_xx = grad_u_x[:, 0]
        v_xx = grad_v_x[:, 0]
        f_u = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v
        f_v = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u

        return stack([f_u, f_v], 1)  # concatenate real part and imaginary part


    pde_loss = pf.PDELoss(geometry, schroedinger1d, name='1D Schrodinger')
    model = pf.models.MLP(input_size=2,
                          output_size=2,
                          hidden_size=args.hidden_size,
                          num_hidden=args.num_hidden,
                          lb=lb,
                          ub=ub)

    logger = pf.WandbLogger('1D Schrödinger Equation', args, 'aipp')
    pinn = pf.PINN(model, 2, 2, pde_loss, initial_condition, [periodic_bc_u,
                                                              periodic_bc_v,
                                                              periodic_bc_u_x,
                                                              periodic_bc_v_x], use_gpu=True)
    pinn.fit(args.num_epochs, checkpoint_path='checkpoint.pt',
             restart=True, logger=logger, activate_annealing=args.annealing, annealing_cycle=args.annealing_cycle,
             writing_cycle=500,
             track_gradient=args.track_gradient)
    pinn.load_model('best_model_pinn.pt')

    # Plotting
    data = scipy.io.loadmat('NLS.mat')
    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)
    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact_u.T.flatten()[:, None]
    v_star = Exact_v.T.flatten()[:, None]
    h_star = Exact_h.T.flatten()[:, None]

    pred = model(Tensor(X_star).cuda())
    pred_u = pred[:, 0].detach().cpu().numpy()
    pred_v = pred[:, 1].detach().cpu().numpy()
    H_pred = np.sqrt(pred_u ** 2 + pred_v**2)
    H_pred = H_pred.reshape(X.shape)
    plt.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    plt.colorbar()
    plt.show()
