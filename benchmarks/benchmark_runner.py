import argparse
import importlib
import numpy as np
import torch
import matplotlib.pyplot as plt

from NeuralSolvers.pinn.datasets import BoundaryConditionDataset1D
from configs import CONFIGS, MODELS, SYSTEM, PDE_FUNCTIONS
import NeuralSolvers as nsolv

class InitialConditionDataset(torch.utils.data.Dataset):
    """Generalized Initial Condition Dataset."""

    def __init__(self, n0, initial_func, domain, device='cpu'):
        x = np.linspace(domain[0][0], domain[1][0], n0)[:, None]
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

    device = SYSTEM['device']
    boundary_conditions = config["boundary_conditions"]
    pinn_boundary_conditions = []
    for bc in boundary_conditions:
        bc_vals = boundary_conditions[bc]
        bc_dataset = BoundaryConditionDataset1D(nb=bc_vals['nb'], is_lower=bc_vals['is_lower'],
                                                DOMAIN_UPPER_BOUND=domain[1], DOMAIN_LOWER_BOUND=domain[0],
                                                device=device)
        bc = nsolv.pinn.datasets.DirichletBC(bc_vals['func'], bc_dataset, name=bc)
        pinn_boundary_conditions.append(bc)

    if "custom_dataset" in config["initial_condition"]:
        module_name, class_name = config["initial_condition"]["custom_dataset"].rsplit(".", 1)
        module = importlib.import_module(module_name)
        CustomDataset = getattr(module, class_name)
        ic_dataset = CustomDataset(**config["initial_condition"]["parameters"], device=device)
    else:
        # Use the default InitialConditionDataset
        ic_dataset = InitialConditionDataset(
            n0=config["initial_condition"]['n0'],
            initial_func=config["initial_condition"]['u0'],
            domain=domain,
            device=device
        )

    initial_condition = nsolv.pinn.datasets.InitialCondition(ic_dataset, name="IC")


    # Use the PDE closure with params
    pde_function = PDE_FUNCTIONS[config["pde_function"]](config["pde_parameters"])

    # Geometry and PDE Loss
    pde_loss = nsolv.pinn.PDELoss(
        nsolv.NDCube(domain[0], domain[1], config["num_collocation_points"], config["num_collocation_points"],
                    nsolv.samplers.LHSSampler(), device=device),
                    pde_function,
                    name = "PDE"
    )

    # Model Arguments
    model_args = config["model_args"]
    model_args.update({
        "lb": domain[0],
        "ub": domain[1],
        "device": device
    })

    # Load the selected model
    model = load_model(model_name, model_args)

    # Initialize PINN
    return nsolv.PINN(model, 2, 1, pde_loss,
                      nsolv.pinn.datasets.InitialCondition(ic_dataset, name = "IC"),
                      pinn_boundary_conditions, device=device
                      )

def train_and_benchmark(system_name, model_name, num_epochs=1000):
    """
    Train and benchmark the PINN.

    Args:
        system_name (str): Name of the PDE system.
        model_name (str): Name of the model architecture.
        num_epochs (int): Number of training epochs.
    """
    config = CONFIGS[system_name]

    pinn = setup_pinn(system_name, model_name)
    pinn.fit(num_epochs, pretraining=config["initial_condition"]["pretrain"],
             lbfgs_finetuning=False)
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
                        choices=["burgers", "heat", "schrodinger", "wave"],
                        help="PDE system to solve: burgers, heat, schrodinger,  wave,")
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