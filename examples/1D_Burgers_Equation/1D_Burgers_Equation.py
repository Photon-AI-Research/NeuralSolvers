import sys

import numpy
import numpy as np
import scipy.io
import torch
from torch import Tensor, ones, stack, load
from torch.autograd import grad
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import NeuralSolvers as nsolv



class InitialConditionDataset(Dataset):

    def __init__(self, n0):
        """
        Constructor of the boundary condition dataset

        Args:
          n0 (int)
        """
        super(type(self)).__init__()
        data = scipy.io.loadmat('burgers_shock.mat')

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
        self.X_u_train = X_u_train[idx, :]
        self.u_train = u_train[idx, :]

    def __len__(self):
        """
        There exists no batch processing. So the size is 1
        """
        return 1

    def __getitem__(self, idx):
        x = self.X_u_train
        y = self.u_train

        return Tensor(x).float(), Tensor(y).float()



if __name__ == "__main__":
    # Domain bounds
    lb = np.array([-1, 0.0])
    ub = np.array([1.0, 1.0])
    
    nu = 0.01 / np.pi
    noise = 0.0

    N_u = 100
    N_f = 10000

    # initial condition
    ic_dataset = InitialConditionDataset(n0=N_u)
    initial_condition = nsolv.InitialCondition(ic_dataset, name='Initial condition')

    #sampler
    sampler = nsolv.LHSSampler()
    #sampler = pf.RandomSampler()
    
    # geometry
    geometry = nsolv.NDCube(lb, ub, N_f, N_f, sampler)

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

    pde_loss = nsolv.PDELoss(geometry, burger1D, name='1D Burgers equation')
    # create model
    model = nsolv.models.MLP(input_size=2, output_size=1,
                             hidden_size=40, num_hidden=8, lb=lb, ub=ub, activation=torch.tanh)
    # create PINN instance
    pinn = nsolv.PINN(model, 2, 1, pde_loss, initial_condition, [], use_gpu=False)

    #logger = nsolv.WandbLogger("1D Burgers equation pinn", args = {})
    logger = None

    # train pinn
    pinn.fit(50000, checkpoint_path='checkpoint.pt', restart=True, logger=logger, lbfgs_finetuning=False, writing_cycle=1000)



    # ========Plotting========

    pinn.load_model('best_model_pinn.pt')  # load best PINN model for plotting
    data = scipy.io.loadmat('burgers_shock.mat')

    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # Domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    pred = pinn(Tensor(X_star).cuda())
    pred = pred.detach().cpu().numpy()
    pred = pred.reshape(X.shape)
    plt.imshow(pred.T, interpolation='nearest', cmap='rainbow',
               extent=[t.min(), t.max(), x.min(), x.max()],
               origin='lower', aspect='auto')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.title(r"$u(x,t)$")
    plt.colorbar()
    plt.show()

