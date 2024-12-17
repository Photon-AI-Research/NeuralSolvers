import numpy as np

# Configuration for PDE systems
CONFIGS = {
    "burgers": {
        "name": "Burgers Equation",
        "domain": [[-1, 0.0], [1.0, 1.0]],  # Spatial and temporal bounds
        "initial_condition": lambda x: -np.sin(np.pi * x),  # u(x, t=0)
        "pde_function": "burgers1D",  # Identifier for the PDE function
        "parameters": {
            "viscosity": 0.01 / np.pi  # Burgers' equation viscosity
        }
    },
    "wave": {
        "name": "Wave Equation",
        "domain": [[-1, 0.0], [1.0, 1.0]],  # Spatial and temporal bounds
        "initial_condition": lambda x: np.sin(np.pi * x),  # u(x, t=0)
        "pde_function": "wave1D",  # Identifier for the PDE function
        "parameters": {
            "wave_speed": 1.0  # Wave propagation speed
        }
    },
    "schrodinger": {
        "name": "Schrödinger Equation",
        "domain": [[-5, 0.0], [5.0, np.pi / 2]],  # Spatial and temporal bounds
        "initial_condition": lambda x: np.exp(-x**2),  # Gaussian initial condition
        "pde_function": "schrodinger1D",  # Identifier for the PDE function
        "parameters": {
            "potential": "harmonic"  # Example: harmonic potential
        }
    }
}

# Available models: mapping of model names to their paths and class names
MODELS = {
    "MLP": "nsolv.models.mlp.MLP",
    "ModulatedMLP": "nsolv.models.modulated_mlp.ModulatedMLP",
}