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
        t = np.linspace(0,max_t,200).flatten()[:, None]
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
        t = np.linspace(0,max_t,200).flatten()[:, None]
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

        L=1               
        c=1               
        alpha = (c*np.pi/L)**2
        max_t = 2
        max_x = L

        t = np.zeros(200).flatten()[:, None]
        x = np.linspace(0,max_x,200).flatten()[:, None]

        U=(np.exp(-(alpha)*t))*np.sin(np.pi*x/L)
        u=U.flatten()[:, None]

        idx_x = np.random.choice(x.shape[0], n0, replace=False)
        self.x = x[idx_x,:]
        self.u = u[idx_x,:]
        self.t = t[idx_x,:]

    def __len__(self):
        """
        There exists no batch processing. So the size is 1
        """
        return 1

    def __getitem__(self, idx):
        x = np.concatenate([self.x, self.t], axis=1)
        y = np.concatenate([self.u], axis=1)
        return Tensor(x).float(), Tensor(y).float()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=10000, help='Number of training iterations')
    parser.add_argument('--n0', dest='n0', type=int, default=50, help='Number of input points for initial condition')
    parser.add_argument('--nb', dest='nb', type=int, default=50, help='Number of input points for boundary condition')
    parser.add_argument('--nf', dest='nf', type=int, default=20000, help='Number of input points for pde loss')
    parser.add_argument('--ns', dest='ns', type=int, default=2000, help='Number of seed points')
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
        return  torch.zeros_like(x)[:,0].reshape(-1,1)
    
    dirichlet_bc_u_lb = pf.DirichletBC(func, bc_datasetlb, name= 'ulb dirichlet boundary condition')
    dirichlet_bc_u_ub = pf.DirichletBC(func, bc_datasetub, name= 'uub dirichlet boundary condition')


    def heat1d(x, u):

        grads = ones(u.shape, device=u.device) # move to the same device as prediction
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
    
    # geometry of the domain
    geometry = pf.Geometry(lb, ub)

    # sampler
    sampler = pf.Sampler(geometry,num_points=args.nf, ns=args.ns, sampler ='adaptive')

    # pde loss
    pde_loss = pf.PDELossAdaptive(geometry, heat1d, sampler, name='1D Heat' )
    
    # create model
    model = pf.models.MLP(input_size=2,
                          output_size=1,
                          hidden_size=args.hidden_size,
                          num_hidden=args.num_hidden,
                          lb=lb,
                          ub=ub)
    
    # create PINN instance
    pinn = pf.PINN(model, 2, 1, pde_loss, initial_condition, [dirichlet_bc_u_lb,dirichlet_bc_u_ub], use_gpu=True)
    
    logger = pf.WandbLogger("1D Heat equation pinn",args)
    
    # train pinn
    pinn.fit(args.num_epochs, checkpoint_path='checkpoint.pt', restart=True, logger=logger,lbfgs_finetuning=False, pretraining = True)
    pinn.load_model('best_model_pinn.pt')

    #Plotting
    max_t = 2
    max_x = 1

    t = np.linspace(0,max_t,200).flatten()[:, None]
    x = np.linspace(0,max_x,200).flatten()[:, None]
    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    pred = pinn(Tensor(X_star).cuda())
    pred_u = pred.detach().cpu().numpy()
    
    H_pred = pred_u.reshape(X.shape)
    plt.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu',
                extent= [lb[1], ub[1], lb[0], ub[0]],
                origin='lower', aspect='auto')
    plt.ylabel('x (cm)')
    plt.xlabel('t (seconds)')
    plt.colorbar().set_label('Temperature (°C)')
    plt.show()




# Analytical Solution

L=1
c=1
max_t = 2
alpha = (c*np.pi/L)**2

# Domain bounds
lb = np.array([0, 0.0])
ub = np.array([L, max_t])

t = np.linspace(0, max_t, 200)
x = np.linspace(0, L, 200)
X, T = np.meshgrid(x, t)
X = X.reshape(-1,1)
T = T.reshape(-1,1)

U=(np.exp(-alpha*T))*np.sin(np.pi*X/L)
U = U.reshape(200,200)

plt.imshow(U.T, interpolation='nearest', cmap='YlGnBu',
                extent= [lb[1], ub[1], lb[0], ub[0]],
                origin='lower', aspect='auto')

plt.ylabel('x (cm)')
plt.xlabel('t (seconds)')
plt.axis()
plt.colorbar().set_label('Temperature (°C)')
plt.show()




#PDEloss plot
L=1
c=1
max_t = 2
alpha = (c*np.pi/L)**2

# Domain bounds
lb = np.array([0, 0.0])
ub = np.array([L, max_t])

t = np.linspace(0, max_t, 200)
x = np.linspace(0, L, 200)
X, T = np.meshgrid(x, t)
X = X.reshape(-1,1)
T = T.reshape(-1,1)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

X_star = torch.tensor(X_star).float().cuda()
X_star.requires_grad = True
plt.figure(figsize=(16,9))
pred = pinn(X_star)

F_u = heat1d(X_star, pred)
F_u = F_u.detach().cpu().numpy()
F_u = F_u.reshape(200,200)


plt.title('PDE residual')
plt.ylabel('x (cm)')
plt.xlabel('t (seconds)')
plt.imshow(F_u.T, cmap='jet', aspect='auto', extent= [lb[1], ub[1], lb[0], ub[0]],
                origin='lower')
plt.colorbar().set_label('F_u')




# PINN vs analytical solution at t_idx=0
plt.plot(x, H_pred[0,:], '--')
plt.plot(x, U[0,:], '-')
plt.title('PINN vs analytical solution at t_idx=0 ({} s)'.format(t[0]))
plt.legend(['PINN', 'Analytical solution'])
plt.xlabel('x / cm')
plt.ylabel('Temperature / °C')
plt.show()

mae = np.sum(np.abs(H_pred[0,:]- U[0,:]).mean(axis=None))
print('MAE:', mae)

mse = ((U[0,:] - H_pred[0,:])**2).mean(axis=None)
print('MSE:', mse)

rel_error = np.linalg.norm(H_pred[0,:]- U[0,:]) / np.linalg.norm(U[0,:])
print('Relative error (%):', rel_error*100)




# PINN vs analytical solution at t_idx = 50
plt.plot(x, H_pred[50,:], '--')
plt.plot(x, U[50,:], '-')
plt.title('PINN vs analytical solution at t_idx=50 ({} s)'.format(t[50]))
plt.legend(['PINN', 'Analytical solution'])
plt.xlabel('x / cm')
plt.ylabel('Temperature / °C')
plt.show()

mae = np.sum(np.abs(H_pred[50,:]- U[50,:]).mean(axis=None))
print('MAE:', mae)

mse = ((U[50,:] - H_pred[50,:])**2).mean(axis=None)
print('MSE:', mse)

rel_error = np.linalg.norm(H_pred[50,:]- U[50,:]) / np.linalg.norm(U[50,:])
print('Relative error (%):', rel_error*100)




# PINN vs analytical solution at t_idx = 100
plt.plot(x, H_pred[100,:], '--')
plt.plot(x, U[100,:], '-')
plt.title('PINN vs analytical solution at t_idx=100 ({} s)'.format(t[100]))
plt.legend(['PINN', 'Analytical solution'])
plt.xlabel('x / cm')
plt.ylabel('Temperature / °C')
plt.show()

mae = np.sum(np.abs(H_pred[100,:]- U[100,:]).mean(axis=None))
print('MAE:', mae)

mse = ((U[100,:] - H_pred[100,:])**2).mean(axis=None)
print('MSE:', mse)

rel_error = np.linalg.norm(H_pred[100,:]- U[100,:]) / np.linalg.norm(U[100,:])
print('Relative error (%):', rel_error*100)




# PINN vs analytical solution at t_idx=199
plt.plot(x, H_pred[199,:], '--')
plt.plot(x, U[199,:], '-')
plt.title('PINN vs analytical solution at t_idx=199 ({}s)'.format(t[199]))
plt.legend(['PINN', 'Analytical solution'])
plt.xlabel('x / cm')
plt.ylabel('Temperature / °C')
plt.show()

mae = np.sum(np.abs(H_pred[199,:]- U[199,:]).mean(axis=None))
print('MAE:', mae)

mse = ((U[199,:] - H_pred[199,:])**2).mean(axis=None)
print('MSE:', mse)

rel_error = np.linalg.norm(H_pred[199,:]- U[199,:]) / np.linalg.norm(U[199,:])
print('Relative error (%):', rel_error*100)




# PINN results at different time points
plt.plot(x, H_pred[0,:], '-')
plt.plot(x, H_pred[50,:], '--')
plt.plot(x, H_pred[199,:], '--')
plt.title('PINN solution at different t values')
plt.legend(['t_idx = 0 ({}s)'.format(t[0]),'t_idx = 50 ({}s)'.format(t[50]), 't_idx = 199 ({}s)'.format(t[199])])
plt.xlabel('position / mm')
plt.ylabel('Temperature / °C')
plt.show()




# Analytical solution at different time points
plt.plot(x, U[0,:], '-')
plt.plot(x, U[50,:], '--')
plt.plot(x, U[199,:], '--')
plt.title('Analytical solution at different t values')
plt.legend(['t_idx = 0 ({}s)'.format(t[0]),'t_idx = 50 ({}s)'.format(t[50]), 't_idx = 199 ({}s)'.format(t[199])])
plt.xlabel('position / mm')
plt.ylabel('Temperature / °C')
plt.show()
