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
            "pretrain": True
        },
        "boundary_conditions": {
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
    },

    "schrodinger": {
        "name": "Schrödinger Equation",
        "domain": np.array([[-5, 0.0], [5.0, np.pi / 2]]),  # Spatial and temporal bounds
        "initial_condition": lambda x: np.exp(-x**2),  # Gaussian initial condition
        "pde_function": "schrodinger1D",  # Identifier for the PDE function
        "parameters": {
            "potential": "harmonic"  # Example: harmonic potential
        },
        "model_args": {
            "input_size": 2,
            "output_size": 2,  # Schrödinger equation typically has complex output
            "hidden_size": 40,
            "num_hidden": 8
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
    }
}

# Available models: mapping of model names to their paths and class names
MODELS = {
    "MLP": "NeuralSolvers.models.mlp.MLP",
    "ModulatedMLP": "NeuralSolvers.models.modulated_mlp.ModulatedMLP",
}

# Additional parameters
SYSTEM = {
    "device": "mps:0",
}

PDE_FUNCTIONS = {
    "burgers1D": burgers1D,
    "wave1D": wave1D,
    "schrodinger1D": schrodinger1D,
    "heat1D": heat1D
}