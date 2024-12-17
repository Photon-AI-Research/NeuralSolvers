import argparse
import importlib
import numpy as np
import torch
import matplotlib.pyplot as plt

from configs import CONFIGS, MODELS
from NeuralSolvers import burgers1D, wave1D, schrodinger1D
import NeuralSolvers as nsolv

PDE_FUNCTIONS = {
    "burgers1D": burgers1D,
    "wave1D": wave1D,
    "schrodinger1D": schrodinger1D
}

class InitialConditionDataset(torch.utils.data.Dataset):
    """Generalized Initial Condition Dataset."""

    def __init__(self, n0, initial_func, device='cpu'):
        x = np.linspace(-1, 1, n0)[:, None]
        u0 = initial_func(x)
        self.X_u_train = torch.Tensor(np.hstack((x, np.zeros_like(x)))).to(device)
        self.u_train = torch.Tensor(u0).to(device)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.X_u_train.float(), self.u_train.float()

def load_model(model_name, model_args):
    """
    Dynamically load a model class and return an instance.

    Args:
        model_name (str): Name of the model class to load.
        model_args (dict): Arguments to pass to the model constructor.

    Returns:
        torch.nn.Module: An instance of the selected model.
    """
    module_name, class_name = MODELS[model_name].rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class(**model_args)

def setup_pinn(system_name, model_name="MLP"):
    """
    Set up the PINN for a given PDE system and model.

    Args:
        system_name (str): The name of the PDE system.
        model_name (str): The name of the model architecture.

    Returns:
        nsolv.PINN: Configured PINN instance.
    """
    config = CONFIGS[system_name]
    domain = config["domain"]

    # Dataset and PDE Loss
    ic_dataset = InitialConditionDataset(
        n0=100, initial_func=config["initial_condition"], device="mps"
    )
    pde_loss = nsolv.pinn.PDELoss(
        nsolv.NDCube(domain[0], domain[1], 1000, 1000, nsolv.samplers.LHSSampler(), device="mps"),
        PDE_FUNCTIONS[config["pde_function"]],
        params=config["parameters"]
    )

    # Model Arguments
    model_args = {
        "input_size": 2,
        "output_size": 1,
        "hidden_size": 40,
        "num_hidden": 8,
        "lb": domain[0],
        "ub": domain[1],
        "activation": torch.tanh,
        "device": "mps"
    }

    # Load the selected model
    model = load_model(model_name, model_args)

    # Initialize PINN
    return nsolv.PINN(model, 2, 1, pde_loss, nsolv.pinn.datasets.InitialCondition(ic_dataset))

def train_and_benchmark(system_name, model_name, num_epochs=1000):
    """
    Train and benchmark the PINN.

    Args:
        system_name (str): Name of the PDE system.
        model_name (str): Name of the model architecture.
        num_epochs (int): Number of training epochs.
    """
    pinn = setup_pinn(system_name, model_name)
    pinn.fit(num_epochs)
    print(f"Finished training for {system_name} using {model_name}.")
    return pinn

def plot_pinn_solution(pinn, system_name):
    """
    Plot the solution predicted by the PINN.

    Args:
        pinn: Trained PINN model.
        system_name (str): Name of the PDE system.
    """
    config = CONFIGS[system_name]
    x = np.linspace(config["domain"][0][0], config["domain"][1][0], 100)
    t = np.linspace(config["domain"][0][1], config["domain"][1][1], 100)
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    pred = pinn(torch.Tensor(X_star).to("mps")).detach().cpu().numpy().reshape(X.shape)
    plt.imshow(pred.T, origin='lower', extent=[t.min(), t.max(), x.min(), x.max()])
    plt.title(f"{system_name} Solution")
    plt.colorbar()
    plt.show()

def main():
    # Argument parser to select system and model
    parser = argparse.ArgumentParser(description="PINN Benchmark Runner")
    parser.add_argument("--system", type=str, required=True,
                        choices=["burgers", "wave", "schrodinger"],
                        help="PDE system to solve: burgers, wave, schrodinger")
    parser.add_argument("--model", type=str, required=True,
                        choices=["MLP", "ModulatedMLP"],
                        help="Model architecture to use: MLP, ModulatedMLP, CustomMLP")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs")
    args = parser.parse_args()

    # Train and benchmark the selected system and model
    pinn = train_and_benchmark(system_name=args.system, model_name=args.model, num_epochs=args.epochs)
    plot_pinn_solution(pinn, system_name=args.system)

if __name__ == "__main__":
    main()