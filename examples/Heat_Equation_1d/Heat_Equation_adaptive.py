import numpy as np
import torch
from torch import Tensor, ones
from torch.autograd import grad
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import NeuralSolvers as nsolv

from Heat_Equation import heat1d

# Constants
DEVICE = 'cpu' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 10000
DOMAIN_LOWER_BOUND = np.array([0, 0.0])
DOMAIN_UPPER_BOUND = np.array([1.0, 2.0])
NUM_INITIAL_POINTS = 50
NUM_BOUNDARY_POINTS = 50
NUM_COLLOCATION_POINTS = 20000
NUM_SEED_POINTS = 10000
HIDDEN_SIZE = 100
NUM_HIDDEN = 4

class InitialConditionDataset(Dataset):
    def __init__(self, n0):
        super().__init__()
        L, c = 1, 1
        alpha = (c * np.pi / L) ** 2
        x = np.linspace(0, L, 200).flatten()[:, None]
        t = np.zeros(200).flatten()[:, None]
        u = (np.exp(-alpha * t)) * np.sin(np.pi * x / L)

        idx = np.random.choice(x.shape[0], n0, replace=False)
        self.x, self.u, self.t = x[idx, :], u[idx, :], t[idx, :]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = np.concatenate([self.x, self.t], axis=1)
        y = self.u
        return Tensor(x).float(), Tensor(y).float()

class BoundaryConditionDataset(Dataset):
    def __init__(self, nb, is_lower):
        super().__init__()
        max_t = 2
        t = np.linspace(0, max_t, 200).flatten()[:, None]
        idx_t = np.random.choice(t.shape[0], nb, replace=False)
        tb = t[idx_t, :]
        x_val = DOMAIN_LOWER_BOUND[0] if is_lower else DOMAIN_UPPER_BOUND[0]
        self.x_b = np.concatenate((np.full_like(tb, x_val), tb), 1)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return Tensor(self.x_b).float()

def setup_pinn():
    ic_dataset = InitialConditionDataset(n0=NUM_INITIAL_POINTS)
    initial_condition = nsolv.pinn.datasets.InitialCondition(ic_dataset, name='Initial condition')

    bc_dataset_lb = BoundaryConditionDataset(nb=NUM_BOUNDARY_POINTS, is_lower=True)
    bc_dataset_ub = BoundaryConditionDataset(nb=NUM_BOUNDARY_POINTS, is_lower=False)

    def dirichlet_func(x):
        return torch.zeros_like(x)[:, 0].reshape(-1, 1)

    dirichlet_bc_lb = nsolv.pinn.datasets.DirichletBC(dirichlet_func, bc_dataset_lb, name='Lower dirichlet BC')
    dirichlet_bc_ub = nsolv.pinn.datasets.DirichletBC(dirichlet_func, bc_dataset_ub, name='Upper dirichlet BC')

    model = nsolv.models.MLP(
        input_size=2, output_size=1, device=DEVICE,
        hidden_size=HIDDEN_SIZE, num_hidden=NUM_HIDDEN,
        lb=DOMAIN_LOWER_BOUND, ub=DOMAIN_UPPER_BOUND
    )

    sampler = nsolv.samplers.AdaptiveSampler(NUM_SEED_POINTS, model, heat1d)
    geometry = nsolv.NDCube(DOMAIN_LOWER_BOUND, DOMAIN_UPPER_BOUND, NUM_COLLOCATION_POINTS, NUM_COLLOCATION_POINTS,
                            sampler, device=DEVICE)

    pde_loss = nsolv.pinn.PDELoss(geometry, heat1d, name='1D Heat')

    return nsolv.pinn.PINN(model, 2, 1, pde_loss, initial_condition, [dirichlet_bc_lb, dirichlet_bc_ub], device=DEVICE)

def train_pinn(pinn):
    '''logger = nsolv.WandbLogger("1D Heat equation pinn", {
        "num_epochs": NUM_EPOCHS,
        "hidden_size": HIDDEN_SIZE,
        "num_hidden": NUM_HIDDEN
    })
    '''
    logger = None
    pinn.fit(NUM_EPOCHS, checkpoint_path='checkpoint.pt', restart=True,
             logger=logger, lbfgs_finetuning=False, pretraining=True)

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
    plt.title("PINN Solution: 1D Heat Equation (Adaptive)")
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
    train_pinn(pinn)
    pinn.load_model('best_model_pinn.pt')
    plot_solution(pinn)
    plot_analytical_solution()