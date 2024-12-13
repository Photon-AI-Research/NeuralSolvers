import numpy as np
import torch
from torch import Tensor, ones
from torch.autograd import grad
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import NeuralSolvers as nsolv
import wandb

# Constants
DEVICE = 'cpu' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 10000
DOMAIN_LOWER_BOUND = np.array([0, 0.0])
DOMAIN_UPPER_BOUND = np.array([1.0, 2.0])
NUM_INITIAL_POINTS = 10000
NUM_COLLOCATION_POINTS = 20000

def derivatives(x, u):
    grads = ones(u.shape, device=u.device)
    grad_u = grad(u, x, create_graph=True, grad_outputs=grads)[0]
    u_x, u_t = grad_u[:, 0], grad_u[:, 1]

    grads = ones(u_x.shape, device=u.device)
    u_xx = grad(u_x, x, create_graph=True, grad_outputs=grads)[0][:, 0]

    u_x, u_t, u_xx = [tensor.reshape(-1, 1) for tensor in (u_x, u_t, u_xx)]
    return torch.cat([u_xx, u_t], dim=1)

class HPM_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = torch.nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, derivatives):
        return self.c * derivatives

class InitialConditionDataset(Dataset):
    def __init__(self, n0):
        super().__init__()
        L, c = 1, 1
        alpha = (c * np.pi / L) ** 2
        max_t, max_x = 10, L

        t = np.linspace(0, max_t, 200)
        x = np.linspace(0, max_x, 200)
        X, T = np.meshgrid(x, t, indexing='ij')

        U = (np.exp(-(alpha)*T)) * np.sin(np.pi*X/L)
        U, X, T = [arr.reshape(-1, 1) for arr in (U, X, T)]

        idx = np.random.choice(X.shape[0], n0, replace=False)
        self.x, self.u, self.t = X[idx, :], U[idx, :], T[idx, :]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = np.concatenate([self.x, self.t], axis=1)
        y = self.u
        return Tensor(x).float(), Tensor(y).float()

def setup_pinn(args):
    ic_dataset = InitialConditionDataset(n0=NUM_INITIAL_POINTS)
    initial_condition = nsolv.pinn.datasets.InitialCondition(ic_dataset, name='Interpolation condition')

    geometry = nsolv.NDCube(DOMAIN_LOWER_BOUND, DOMAIN_UPPER_BOUND, NUM_COLLOCATION_POINTS, NUM_COLLOCATION_POINTS,
                            nsolv.samplers.LHSSampler(), device=DEVICE)

    hpm_model = HPM_model()
    #wandb.watch(hpm_model)
    pde_loss = nsolv.pinn.HPMLoss(geometry, "HPM_loss", derivatives, hpm_model)

    model = nsolv.models.MLP(
        input_size=2, output_size=1, device=DEVICE,
        hidden_size=args.hidden_size, num_hidden=args.num_hidden,
        lb=DOMAIN_LOWER_BOUND, ub=DOMAIN_UPPER_BOUND
    )

    return nsolv.pinn.PINN(model, 2, 1, pde_loss, initial_condition, boundary_condition=None, device=DEVICE)

def train_pinn(pinn, args):
    #logger = nsolv.WandbLogger("1D Heat equation inversion", args)
    logger = None
    pinn.fit(args.num_epochs, epochs_pt=200, checkpoint_path=None, restart=True,
             logger=logger, lbfgs_finetuning=False, pretraining=True)

def plot_solution(pinn):
    t = np.linspace(0, DOMAIN_UPPER_BOUND[1], 200).flatten()[:, None]
    x = np.linspace(DOMAIN_LOWER_BOUND[0], DOMAIN_UPPER_BOUND[0], 200).flatten()[:, None]
    X, T = np.meshgrid(x, t, indexing='ij')
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    pred = pinn(Tensor(X_star).to(DEVICE))
    H_pred = pred.detach().cpu().numpy().reshape(X.shape)

    plt.figure(figsize=(10, 8))
    plt.imshow(H_pred, interpolation='nearest', cmap='YlGnBu',
               extent=[DOMAIN_LOWER_BOUND[1], DOMAIN_UPPER_BOUND[1], DOMAIN_LOWER_BOUND[0], DOMAIN_UPPER_BOUND[0]],
               origin='lower', aspect='auto')
    plt.ylabel('x (cm)')
    plt.xlabel('t (seconds)')
    plt.colorbar().set_label('Temperature (°C)')
    plt.title("PINN Solution: 1D Heat Equation Inversion")
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--num_hidden", type=int, default=4)
    args = parser.parse_args()

    pinn = setup_pinn(args)
    train_pinn(pinn, args)
    pinn.load_model('best_model_pinn.pt', 'best_model_hpm.pt')
    plot_solution(pinn)
    plot_analytical_solution()

    # Print the inferred parameter
    print(f"Inferred parameter c: {pinn.pde_loss.hpm_model.c.item()}")
