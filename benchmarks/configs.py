import numpy as np
import torch
from NeuralSolvers import *

# Configuration for PDE systems
CONFIGS = {
    "burgers": {
        "name": "Burgers Equation",
        "domain": np.array([[-1, 0.0], [1.0, 1.0]]),  # Spatial and temporal bounds
        "pde_function": "burgers1D",  # Identifier for the PDE function
        "pde_parameters": {
            "viscosity": 0.01 / np.pi  # Burgers' equation viscosity
        },
        "num_collocation_points": 10000,
        "initial_condition": {
            "custom_dataset": "examples.Burgers_Equation_1d.Burgers_Equation.InitialConditionDataset",
            "parameters": {
                "n0": 100,
                "file_path": "../examples/Burgers_Equation_1d/burgers_shock.mat"
            },
            "pretrain": False
        },
        "boundary_conditions": {
        },
        "model_args": {
            "input_size": 2,
            "output_size": 1,
            "hidden_size": 40,
            "num_hidden": 8,
            "activation": torch.tanh
        }
    },

    "schrodinger": {
        "name": "Schrödinger Equation",
        "domain": np.array([[-5, 0.0], [5.0, np.pi / 2]]),  # Spatial and temporal bounds
        "pde_function": "schrodinger1D",  # Identifier for the PDE function
        "pde_parameters": {
            "potential": "harmonic"  # Example: harmonic potential
        },
        "num_collocation_points": 20000,
        "initial_condition": {
            "custom_dataset": "examples.Schroedinger_1d.Schroedinger.InitialConditionDataset",
            "parameters": {
                "n0": 50,
                "file_path": "../examples/Schroedinger_1d/NLS.mat"
            },
            "pretrain": False
        },
        "boundary_conditions": {
            "u_periodic": {
                "custom_dataset": "examples.Schroedinger_1d.Schroedinger.BoundaryConditionDataset",
                "custom_dataset_parameters": {
                    "nb": 50,
                    "file_path": "../examples/Schroedinger_1d/NLS.mat"
                },
                "custom_boundary": "NeuralSolvers.pinn.datasets.PeriodicBC",
                "custom_boundary_parameters": {
                    "output_dimension": 0,
                    "name": "u periodic"
                },
            },
            "v_periodic": {
                "custom_dataset": "examples.Schroedinger_1d.Schroedinger.BoundaryConditionDataset",
                "custom_dataset_parameters": {
                    "nb": 50,
                    "file_path": "../examples/Schroedinger_1d/NLS.mat"
                },
                "custom_boundary": "NeuralSolvers.pinn.datasets.PeriodicBC",
                "custom_boundary_parameters": {
                    "output_dimension": 1,
                    "name": "v periodic"
                },
            },
            "ux_periodic": {
                "custom_dataset": "examples.Schroedinger_1d.Schroedinger.BoundaryConditionDataset",
                "custom_dataset_parameters": {
                    "nb": 50,
                    "file_path": "../examples/Schroedinger_1d/NLS.mat"
                },
                "custom_boundary": "NeuralSolvers.pinn.datasets.PeriodicBC",
                "custom_boundary_parameters": {
                    "name": "u_x periodic",
                    "degree": 1,
                    "input_dimension": 0,
                    "output_dimension": 0

                },
            },
            "vx_periodic": {
                "custom_dataset": "examples.Schroedinger_1d.Schroedinger.BoundaryConditionDataset",
                "custom_dataset_parameters": {
                    "nb": 50,
                    "file_path": "../examples/Schroedinger_1d/NLS.mat"
                },
                "custom_boundary": "NeuralSolvers.pinn.datasets.PeriodicBC",
                "custom_boundary_parameters": {
                    "name": "v_x periodic",
                    "degree": 1,
                    "input_dimension": 0,
                    "output_dimension": 1
                },
            }
        },
        "model_args": {
            "input_size": 2,
            "output_size": 2,
            "hidden_size": 100,
            "num_hidden": 4,
            "activation": torch.tanh
        }
    },

    "heat": {
        "name": "Heat Equation",
        "domain": np.array([[0.0, 0.0], [1.0, 2.0]]),  # Spatial and temporal bounds
        "pde_function": "heat1D",  # Identifier for the PDE function
        "num_collocation_points": 20000,
        "pde_parameters": {
            "diffusivity": 1.0  # Thermal diffusivity (assumed to be 1 in this case)
        },
        "initial_condition": {
            "u0": lambda x: np.sin(np.pi * x),
            "n0": 50,
            "pretrain": True
        },
        "boundary_conditions": {
            "DC_upper": {
                "nb": 100,
                "func": dirichlet,
                "is_lower": False
            },
            "DC_lower": {
                "nb": 100,
                "func": dirichlet,
                "is_lower": True
            }
        },
        "model_args": {
            "input_size": 2,
            "output_size": 1,
            "hidden_size": 100,
            "num_hidden": 4,
            "activation": torch.tanh
        }
    },

    "wave": {
        "name": "Wave Equation",
        "domain": np.array([[-1, 0.0], [1.0, 1.0]]),  # Spatial and temporal bounds
        "initial_condition": lambda x: np.sin(np.pi * x),  # u(x, t=0)
        "pde_function": "wave1D",  # Identifier for the PDE function
        "parameters": {
            "wave_speed": 1.0  # Wave propagation speed
        },
        "model_args": {
            "input_size": 2,
            "output_size": 1,
            "hidden_size": 40,
            "num_hidden": 8
        }
    }
}

# Available models: mapping of model names to their paths and class names
MODELS = {
    "MLP": "NeuralSolvers.models.mlp.MLP",
    "ModulatedMLP": "NeuralSolvers.models.modulated_mlp.ModulatedMLP",
}

# Additional parameters
SYSTEM = {
    "device": "cuda:0",
}

PDE_FUNCTIONS = {
    "burgers1D": burgers1D,
    "wave1D": wave1D,
    "schrodinger1D": schrodinger1D,
    "heat1D": heat1D
}