import numpy as np
import scipy.io
import torch
from torch import Tensor, ones
from torch.autograd import grad
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import NeuralSolvers as nsolv
from Burgers_Equation import burger1D

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 50000
DOMAIN_LOWER_BOUND = np.array([-1, 0.0])
DOMAIN_UPPER_BOUND = np.array([1.0, 1.0])
VISCOSITY = 0.01 / np.pi
NOISE = 0.0
NUM_INITIAL_POINTS = 100
NUM_COLLOCATION_POINTS = 10000
ADAPTIVE_SAMPLE_SIZE = 5000


class InitialConditionDataset(Dataset):
    def __init__(self, n0, device=DEVICE):
        super().__init__()
        self.device = device
        data = scipy.io.loadmat('burgers_shock.mat')

        t, x = data['t'].flatten()[:, None], data['x'].flatten()[:, None]
        Exact = np.real(data['usol']).T

        X, T = np.meshgrid(x, t)
        X_u_train = np.vstack([
            np.hstack((X[0:1, :].T, T[0:1, :].T)),
            np.hstack((X[:, 0:1], T[:, 0:1])),
            np.hstack((X[:, -1:], T[:, -1:]))
        ])
        u_train = np.vstack([
            Exact[0:1, :].T,
            Exact[:, 0:1],
            Exact[:, -1:]
        ])

        idx = np.random.choice(X_u_train.shape[0], n0, replace=False)
        self.X_u_train = Tensor(X_u_train[idx, :]).to(self.device)
        self.u_train = Tensor(u_train[idx, :]).to(self.device)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.X_u_train.float(), self.u_train.float()


def load_burger_data(file_path='burgers_shock.mat'):
    data = scipy.io.loadmat(file_path)
    t, x = data['t'].flatten()[:, None], data['x'].flatten()[:, None]
    exact_solution = np.real(data['usol']).T
    return t, x, exact_solution


def setup_pinn():
    ic_dataset = InitialConditionDataset(n0=NUM_INITIAL_POINTS, device=DEVICE)
    initial_condition = nsolv.pinn.datasets.InitialCondition(ic_dataset, name='Initial condition')

    model = nsolv.models.MLP(
        input_size=2, output_size=1, device=DEVICE,
        hidden_size=40, num_hidden=8, lb=DOMAIN_LOWER_BOUND, ub=DOMAIN_UPPER_BOUND,
        activation=torch.tanh
    )

    sampler = nsolv.samplers.AdaptiveSampler(ADAPTIVE_SAMPLE_SIZE, model, burger1D)
    geometry = nsolv.NDCube(DOMAIN_LOWER_BOUND, DOMAIN_UPPER_BOUND, NUM_COLLOCATION_POINTS, NUM_COLLOCATION_POINTS,
                            sampler, device=DEVICE)

    pde_loss = nsolv.pinn.PDELoss(geometry, burger1D, name='1D Burgers equation')

    return nsolv.pinn.PINN(model, 2, 1, pde_loss, initial_condition, [], device=DEVICE)


def train_pinn(pinn, num_epochs):
    #logger = nsolv.WandbLogger("1D Burgers equation pinn", {"num_epochs": num_epochs})
    logger = None
    pinn.fit(num_epochs, checkpoint_path='checkpoint.pt', restart=True,
             logger=logger, lbfgs_finetuning=False, writing_cycle=1000)


def plot_solution(pinn, t, x, exact_solution):
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    pred = pinn(Tensor(X_star).to(DEVICE))
    pred = pred.detach().cpu().numpy().reshape(X.shape)

    plt.figure(figsize=(10, 8))
    plt.imshow(pred.T, interpolation='nearest', cmap='rainbow',
               extent=[t.min(), t.max(), x.min(), x.max()],
               origin='lower', aspect='auto')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.title(r"$u(x,t)$ - Adaptive PINN Solution")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    pinn = setup_pinn()
    train_pinn(pinn, NUM_EPOCHS)

    pinn.load_model('best_model_pinn.pt')
    t, x, exact_solution = load_burger_data()
    plot_solution(pinn, t, x, exact_solution)