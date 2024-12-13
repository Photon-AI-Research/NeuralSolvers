import numpy as np
import torch
from torch import Tensor, ones
from torch.autograd import grad
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import NeuralSolvers as nsolv

# Constants
DEVICE = 'cuda'
NUM_EPOCHS = 1000
DOMAIN_LOWER_BOUND = np.array([0, 0.0])
DOMAIN_UPPER_BOUND = np.array([1.0, 2.0])
NUM_INITIAL_POINTS = 50
NUM_BOUNDARY_POINTS = 50
NUM_COLLOCATION_POINTS = 20000


def heat1d(x, u):
    grads = ones(u.shape, device=u.device)
    grad_u = grad(u, x, create_graph=True, grad_outputs=grads)[0]
    u_x, u_t = grad_u[:, 0], grad_u[:, 1]

    grads = ones(u_x.shape, device=u.device)
    u_xx = grad(u_x, x, create_graph=True, grad_outputs=grads)[0][:, 0]

    u_x, u_t, u_xx = [tensor.reshape(-1, 1) for tensor in (u_x, u_t, u_xx)]

    return u_t - u_xx


class InitialConditionDataset(Dataset):
    def __init__(self, n0):
        super().__init__()
        L, c = 1, 1
        alpha = (c * np.pi / L) ** 2
        x = np.linspace(0, L, 200).flatten()[:, None]
        t = np.zeros(200).flatten()[:, None]
        u = (np.exp(-alpha * t)) * np.sin(np.pi * x / L)

        idx = np.random.choice(x.shape[0], n0, replace=False)
        self.x = Tensor(x[idx, :]).float().to(DEVICE)
        self.u = Tensor(u[idx, :]).float().to(DEVICE)
        self.t = Tensor(t[idx, :]).float().to(DEVICE)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = torch.cat([self.x, self.t], dim=1)
        y = self.u
        return x,y


class BoundaryConditionDataset(Dataset):
    def __init__(self, nb, is_lower):
        super().__init__()
        max_t = 2
        t = np.linspace(0, max_t, 200).flatten()[:, None]
        idx_t = np.random.choice(t.shape[0], nb, replace=False)
        tb = t[idx_t, :]
        x_val = DOMAIN_LOWER_BOUND[0] if is_lower else DOMAIN_UPPER_BOUND[0]
        self.x_b = Tensor(np.concatenate((np.full_like(tb, x_val), tb), 1)).float().to(DEVICE)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.x_b


def setup_pinn():
    ic_dataset = InitialConditionDataset(n0=NUM_INITIAL_POINTS)
    initial_condition = nsolv.pinn.datasets.InitialCondition(ic_dataset, name='Initial Condition loss')

    bc_dataset_lb = BoundaryConditionDataset(nb=NUM_BOUNDARY_POINTS, is_lower=True)
    bc_dataset_ub = BoundaryConditionDataset(nb=NUM_BOUNDARY_POINTS, is_lower=False)

    def dirichlet_func(x):
        return torch.zeros_like(x)[:, 0].reshape(-1, 1)

    dirichlet_bc_lb = nsolv.pinn.datasets.DirichletBC(dirichlet_func, bc_dataset_lb, name='Lower dirichlet BC')
    dirichlet_bc_ub = nsolv.pinn.datasets.DirichletBC(dirichlet_func, bc_dataset_ub, name='Upper dirichlet BC')

    geometry = nsolv.NDCube(DOMAIN_LOWER_BOUND, DOMAIN_UPPER_BOUND, NUM_COLLOCATION_POINTS, NUM_COLLOCATION_POINTS,
                            nsolv.samplers.LHSSampler(), device=DEVICE)

    pde_loss = nsolv.pinn.PDELoss(geometry, heat1d, name='PDE loss', weight=1)

    model = nsolv.models.MLP(
        input_size=2, output_size=1, device=DEVICE,
        hidden_size=100, num_hidden=4, lb=DOMAIN_LOWER_BOUND, ub=DOMAIN_UPPER_BOUND,
        activation=torch.tanh
    )

    return nsolv.pinn.PINN(model, 2, 1, pde_loss, initial_condition, [dirichlet_bc_lb, dirichlet_bc_ub], device=DEVICE)


def train_pinn(pinn, num_epochs, logger = None):
    #logger = nsolv.WandbLogger("1D Heat equation pinn", {"num_epochs": num_epochs})
    pinn.fit(num_epochs, checkpoint_path='checkpoint.pt', restart=True, logger=logger,
             lbfgs_finetuning=False, pretraining=True)


def plot_solution(pinn):
    t = np.linspace(0, DOMAIN_UPPER_BOUND[1], 200).flatten()[:, None]
    x = np.linspace(DOMAIN_LOWER_BOUND[0], DOMAIN_UPPER_BOUND[0], 200).flatten()[:, None]
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    pred = pinn(Tensor(X_star).to(DEVICE))
    H_pred = pred.detach().cpu().numpy().reshape(X.shape)

    plt.figure(figsize=(10, 8))
    plt.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu',
               extent=[DOMAIN_LOWER_BOUND[1], DOMAIN_UPPER_BOUND[1], DOMAIN_LOWER_BOUND[0], DOMAIN_UPPER_BOUND[0]],
               origin='lower', aspect='auto')
    plt.ylabel('x (cm)')
    plt.xlabel('t (seconds)')
    plt.colorbar().set_label('Temperature (°C)')
    plt.title("PINN Solution: 1D Heat Equation")
    plt.show()


def plot_analytical_solution():
    L, c = 1, 1
    alpha = (c * np.pi / L) ** 2
    t = np.linspace(0, DOMAIN_UPPER_BOUND[1], 200)
    x = np.linspace(DOMAIN_LOWER_BOUND[0], DOMAIN_UPPER_BOUND[0], 200)
    X, T = np.meshgrid(x, t)
    U = (np.exp(-alpha * T)) * np.sin(np.pi * X / L)

    plt.figure(figsize=(10, 8))
    plt.imshow(U.T, interpolation='nearest', cmap='YlGnBu',
               extent=[DOMAIN_LOWER_BOUND[1], DOMAIN_UPPER_BOUND[1], DOMAIN_LOWER_BOUND[0], DOMAIN_UPPER_BOUND[0]],
               origin='lower', aspect='auto')
    plt.ylabel('x (cm)')
    plt.xlabel('t (seconds)')
    plt.colorbar().set_label('Temperature (°C)')
    plt.title("Analytical Solution: 1D Heat Equation")
    plt.show()


if __name__ == "__main__":
    pinn = setup_pinn()
    train_pinn(pinn, NUM_EPOCHS)
    plot_solution(pinn)
    plot_analytical_solution()