import numpy as np
import scipy.io
import torch
from torch import Tensor, ones, stack
from torch.autograd import grad
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import NeuralSolvers as nsolv

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 1000 # 10000
DOMAIN_LOWER_BOUND = np.array([-5.0, 0.0])
DOMAIN_UPPER_BOUND = np.array([5.0, np.pi / 2])
NUM_INITIAL_POINTS = 50
NUM_BOUNDARY_POINTS = 50
NUM_COLLOCATION_POINTS = 20000


def schroedinger1d(x, u):
    u_real, u_imag = u[:, 0], u[:, 1]

    grads = ones(u_real.shape, device=u.device)
    grad_u_real = grad(u_real, x, create_graph=True, grad_outputs=grads)[0]
    grad_u_imag = grad(u_imag, x, create_graph=True, grad_outputs=grads)[0]

    u_real_x, u_real_t = grad_u_real[:, 0], grad_u_real[:, 1]
    u_imag_x, u_imag_t = grad_u_imag[:, 0], grad_u_imag[:, 1]

    u_real_xx = grad(u_real_x, x, create_graph=True, grad_outputs=grads)[0][:, 0]
    u_imag_xx = grad(u_imag_x, x, create_graph=True, grad_outputs=grads)[0][:, 0]

    f_real = u_real_t + 0.5 * u_imag_xx + (u_real ** 2 + u_imag ** 2) * u_imag
    f_imag = u_imag_t - 0.5 * u_real_xx - (u_real ** 2 + u_imag ** 2) * u_real

    return stack([f_real, f_imag], 1)


class InitialConditionDataset(Dataset):
    def __init__(self, n0):
        super().__init__()
        data = scipy.io.loadmat('NLS.mat')
        x = data['x'].flatten()[:, None]
        Exact = data['uu']
        Exact_u = np.real(Exact)
        Exact_v = np.imag(Exact)
        idx_x = np.random.choice(x.shape[0], n0, replace=False)
        self.x = x[idx_x, :]
        self.u = Exact_u[idx_x, 0:1]
        self.v = Exact_v[idx_x, 0:1]
        self.t = np.zeros(self.x.shape)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = np.concatenate([self.x, self.t], axis=1)
        y = np.concatenate([self.u, self.v], axis=1)
        return Tensor(x).float(), Tensor(y).float()


class BoundaryConditionDataset(Dataset):
    def __init__(self, nb):
        super().__init__()
        data = scipy.io.loadmat('NLS.mat')
        t = data['tt'].flatten()[:, None]
        idx_t = np.random.choice(t.shape[0], nb, replace=False)
        tb = t[idx_t, :]
        self.x_lb = np.concatenate((np.full_like(tb, DOMAIN_LOWER_BOUND[0]), tb), 1)
        self.x_ub = np.concatenate((np.full_like(tb, DOMAIN_UPPER_BOUND[0]), tb), 1)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return Tensor(self.x_lb).float(), Tensor(self.x_ub).float()


def setup_pinn():
    ic_dataset = InitialConditionDataset(n0=NUM_INITIAL_POINTS)
    initial_condition = nsolv.InitialCondition(ic_dataset, name='Initial condition')

    bc_dataset = BoundaryConditionDataset(nb=NUM_BOUNDARY_POINTS)
    periodic_bc_u = nsolv.PeriodicBC(bc_dataset, 0, "u periodic boundary condition")
    periodic_bc_v = nsolv.PeriodicBC(bc_dataset, 1, "v periodic boundary condition")
    periodic_bc_u_x = nsolv.PeriodicBC(bc_dataset, 0, "u_x periodic boundary condition", 1, 0)
    periodic_bc_v_x = nsolv.PeriodicBC(bc_dataset, 1, "v_x periodic boundary condition", 1, 0)

    geometry = nsolv.NDCube(DOMAIN_LOWER_BOUND, DOMAIN_UPPER_BOUND, NUM_COLLOCATION_POINTS, NUM_COLLOCATION_POINTS,
                            nsolv.LHSSampler(), device=DEVICE)

    pde_loss = nsolv.PDELoss(geometry, schroedinger1d, name='1D Schrodinger')

    model = nsolv.models.MLP(
        input_size=2, output_size=2, device=DEVICE,
        hidden_size=100, num_hidden=4, lb=DOMAIN_LOWER_BOUND, ub=DOMAIN_UPPER_BOUND,
        activation=torch.tanh
    )

    return nsolv.PINN(model, 2, 2, pde_loss, initial_condition,
                      [periodic_bc_u, periodic_bc_v, periodic_bc_u_x, periodic_bc_v_x], device=DEVICE)


def train_pinn(pinn, num_epochs):
    #logger = nsolv.WandbLogger('1D Schrödinger Equation', {"num_epochs": num_epochs})
    logger = None
    pinn.fit(num_epochs, checkpoint_path='checkpoint.pt', restart=True, logger=logger,
             lbfgs_finetuning=False, writing_cycle=500)


def plot_solution(pinn):
    data = scipy.io.loadmat('NLS.mat')
    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    pred = pinn(Tensor(X_star).to(DEVICE))
    pred_u = pred[:, 0].detach().cpu().numpy()
    pred_v = pred[:, 1].detach().cpu().numpy()
    H_pred = np.sqrt(pred_u ** 2 + pred_v ** 2).reshape(X.shape)

    plt.figure(figsize=(10, 8))
    plt.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu',
               extent=[DOMAIN_LOWER_BOUND[1], DOMAIN_UPPER_BOUND[1], DOMAIN_LOWER_BOUND[0], DOMAIN_UPPER_BOUND[0]],
               origin='lower', aspect='auto')
    plt.ylabel('x')
    plt.xlabel('t')
    plt.colorbar().set_label('|ψ|')
    plt.title("PINN Solution: 1D Schrödinger Equation")
    plt.show()


def plot_exact_solution():
    data = scipy.io.loadmat('NLS.mat')
    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(Exact_h.T, interpolation='nearest', cmap='YlGnBu',
               extent=[DOMAIN_LOWER_BOUND[1], DOMAIN_UPPER_BOUND[1], DOMAIN_LOWER_BOUND[0], DOMAIN_UPPER_BOUND[0]],
               origin='lower', aspect='auto')
    plt.ylabel('x')
    plt.xlabel('t')
    plt.colorbar().set_label('|ψ|')
    plt.title("Exact Solution: 1D Schrödinger Equation")
    plt.show()


def compare_solutions(pinn):
    data = scipy.io.loadmat('NLS.mat')
    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)

    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    pred = pinn(Tensor(X_star).to(DEVICE))
    pred_u = pred[:, 0].detach().cpu().numpy()
    pred_v = pred[:, 1].detach().cpu().numpy()
    H_pred = np.sqrt(pred_u ** 2 + pred_v ** 2).reshape(X.shape)

    error = np.abs(H_pred - Exact_h.T)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(Exact_h.T, interpolation='nearest', cmap='YlGnBu',
               extent=[DOMAIN_LOWER_BOUND[1], DOMAIN_UPPER_BOUND[1], DOMAIN_LOWER_BOUND[0], DOMAIN_UPPER_BOUND[0]],
               origin='lower', aspect='auto')
    plt.ylabel('x')
    plt.xlabel('t')
    plt.colorbar().set_label('|ψ|')
    plt.title("Exact Solution")

    plt.subplot(1, 3, 2)
    plt.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu',
               extent=[DOMAIN_LOWER_BOUND[1], DOMAIN_UPPER_BOUND[1], DOMAIN_LOWER_BOUND[0], DOMAIN_UPPER_BOUND[0]],
               origin='lower', aspect='auto')
    plt.ylabel('x')
    plt.xlabel('t')
    plt.colorbar().set_label('|ψ|')
    plt.title("PINN Solution")

    plt.subplot(1, 3, 3)
    plt.imshow(error.T, interpolation='nearest', cmap='hot',
               extent=[DOMAIN_LOWER_BOUND[1], DOMAIN_UPPER_BOUND[1], DOMAIN_LOWER_BOUND[0], DOMAIN_UPPER_BOUND[0]],
               origin='lower', aspect='auto')
    plt.ylabel('x')
    plt.xlabel('t')
    plt.colorbar().set_label('Error')
    plt.title("Absolute Error")

    plt.tight_layout()
    plt.show()

    print(f"Mean Absolute Error: {np.mean(error)}")
    print(f"Max Absolute Error: {np.max(error)}")


if __name__ == "__main__":
    pinn = setup_pinn()
    train_pinn(pinn, NUM_EPOCHS)
    plot_solution(pinn)
    plot_exact_solution()
    compare_solutions(pinn)