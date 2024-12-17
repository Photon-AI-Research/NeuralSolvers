from examples.Heat_Equation_1d import Heat_Equation
from examples.Schroedinger_1d import Schroedinger
from examples.Burgers_Equation_1d import Burgers_Equation
from NeuralSolvers.loggers.Python_Logger import PythonLogger
from NeuralSolvers.models import ModulatedMLP
import numpy as np
import torch
import scipy

DEVICE = 'mps'
NUM_EPOCHS = 1000  # 50000
DOMAIN_LOWER_BOUND = np.array([-1, 0.0])
DOMAIN_UPPER_BOUND = np.array([1.0, 1.0])
VISCOSITY = 0.01 / np.pi
NOISE = 0.0
NUM_INITIAL_POINTS = 100
NUM_COLLOCATION_POINTS = 10000


def run_benchmarks():
    plot_results_debug = True

    # the loggers store final loss for reasons of comparison
    burger_log = PythonLogger()
    heat_log = PythonLogger()
    schrodinger_log = PythonLogger()
    burger_log_mod = PythonLogger()
    heat_log_mod = PythonLogger()
    schrodinger_log_mod = PythonLogger()

    '''
    Burgers Equation
    '''
    print("*** Burgers Equation ***")

    DEVICE = 'mps'
    DOMAIN_LOWER_BOUND = np.array([-1, 0.0])
    DOMAIN_UPPER_BOUND = np.array([1.0, 1.0])
    model = ModulatedMLP(
        input_size=2, output_size=1, device=DEVICE,
        hidden_size=40, num_hidden=8, lb=DOMAIN_LOWER_BOUND, ub=DOMAIN_UPPER_BOUND,
        activation=torch.tanh
    )

    # premodulation (cuda)
    #[2024-12-14 23:03:02.502914]:Epoch 1000/1000 | PINN Loss 0.0066323378 | Initial Condition loss: 0.003978 | PDE loss: 0.002655 | Epoch Duration 0.02852

    ## vanilla PINN
    # [2024-12-14 21:57:03.422603]:Epoch 1000/1000 | PINN Loss 0.0302316695 | Initial Condition loss: 0.020797 | PDE loss: 0.009435 | Epoch Duration 0.00928

    pinn = Burgers_Equation.setup_pinn(model=model, file_path = '../examples/Burgers_Equation_1d/burgers_shock.mat')
    Burgers_Equation.train_pinn(pinn, Burgers_Equation.NUM_EPOCHS, logger=burger_log_mod)

    pinn = Burgers_Equation.setup_pinn(model=None, file_path = '../examples/Burgers_Equation_1d/burgers_shock.mat')
    Burgers_Equation.train_pinn(pinn, Burgers_Equation.NUM_EPOCHS, logger=burger_log)

    if(plot_results_debug):
        t, x, exact_solution = Burgers_Equation.load_burger_data(file_path = '../examples/Burgers_Equation_1d/burgers_shock.mat')
        Burgers_Equation.plot_solution(pinn, t, x, exact_solution)

    '''
    Heat Equation
    '''
    print("*** Heat Equation ***")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'
    DOMAIN_LOWER_BOUND = np.array([0, 0.0])
    DOMAIN_UPPER_BOUND = np.array([1.0, 2.0])

    model = ModulatedMLP(
        input_size=2, output_size=1, device=DEVICE,
        hidden_size=100, num_hidden=4, lb=DOMAIN_LOWER_BOUND, ub=DOMAIN_UPPER_BOUND,
        activation=torch.tanh
    )
    model = None

    # premodulation (mps)
    #[2024-12-15 17:58:13.849840]:Epoch 1000/1000 | PINN Loss 0.0001808163 | Initial Condition loss loss: 0.000020 | PDE loss loss: 0.000151 | Lower dirichlet BC loss: 0.000005 | Upper dirichlet BC loss: 0.000004 | Epoch Duration 0.25264

    # vanilla PINN
# [2024-12-15 19:35:19.770822]:Epoch 1000/1000 | PINN Loss 0.0004855946 | Initial Condition loss loss: 0.000077 | PDE loss loss: 0.000373 | Lower dirichlet BC loss: 0.000014 | Upper dirichlet BC loss: 0.000022 | Epoch Duration 0.07480

    pinn = Heat_Equation.setup_pinn(model=model)
    Heat_Equation.train_pinn(pinn, Heat_Equation.NUM_EPOCHS, logger=heat_log_mod)

    pinn = Heat_Equation.setup_pinn(model=None)
    Heat_Equation.train_pinn(pinn, Heat_Equation.NUM_EPOCHS, logger=heat_log)

    if (plot_results_debug):
        Heat_Equation.plot_solution(pinn)
        Heat_Equation.plot_analytical_solution()

    '''
    Schroedinger Equation
    '''
    print("*** Schroedinger Equation ***")
    # premodulation (mps)
    #[2024-12-15 18:12:43.802903]:Epoch 1000/1000 | PINN Loss 0.0008542041 | Initial Condition loss loss: 0.000132 | PDE loss loss: 0.000715 | u periodic boundary condition loss: 0.000005 | v periodic boundary condition loss: 0.000000 | u_x periodic boundary condition loss: 0.000002 | v_x periodic boundary condition loss: 0.000001 | Epoch Duration 0.50807

    # vanilla PINN
    # [2024-12-15 19:37:46.156288]:Epoch 1000/1000 | PINN Loss 0.0211467128 | Initial Condition loss loss: 0.011177 | PDE loss loss: 0.009758 | u periodic boundary condition loss: 0.000022 | v periodic boundary condition loss: 0.000160 | u_x periodic boundary condition loss: 0.000020 | v_x periodic boundary condition loss: 0.000010 | Epoch Duration 0.14279

    file_path = '../examples/Schroedinger_1d/NLS.mat'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'
    DOMAIN_LOWER_BOUND = np.array([-5.0, 0.0])
    DOMAIN_UPPER_BOUND = np.array([5.0, np.pi / 2])
    model = ModulatedMLP(
        input_size=2, output_size=2, device=DEVICE,
        hidden_size=100, num_hidden=4, lb=DOMAIN_LOWER_BOUND, ub=DOMAIN_UPPER_BOUND,
        activation=torch.tanh
    )
    pinn = Schroedinger.setup_pinn(file_path=file_path, model=model)
    Schroedinger.train_pinn(pinn, Schroedinger.NUM_EPOCHS, logger=schrodinger_log_mod)

    pinn = Schroedinger.setup_pinn(file_path=file_path, model=None)
    Schroedinger.train_pinn(pinn, Schroedinger.NUM_EPOCHS, logger=schrodinger_log)

    if (plot_results_debug):
        Schroedinger.plot_solution(pinn, file_path=file_path)
        Schroedinger.plot_exact_solution(file_path=file_path)
        Schroedinger.compare_solutions(pinn, file_path=file_path)

    print("*** Burgers Equation ***")
    print(", ".join([f"{key}: {value}" for key, value in burger_log.loss_history.items()]))
    print(", ".join([f"{key}: {value}" for key, value in burger_log_mod.loss_history.items()]))

    print("*** Heat Equation ***")
    print(", ".join([f"{key}: {value}" for key, value in heat_log.loss_history.items()]))
    print(", ".join([f"{key}: {value}" for key, value in heat_log_mod.loss_history.items()]))

    print("*** Schroedinger Equation ***")
    print(", ".join([f"{key}: {value}" for key, value in schrodinger_log.loss_history.items()]))
    print(", ".join([f"{key}: {value}" for key, value in schrodinger_log_mod.loss_history.items()]))



if __name__ == "__main__":
    run_benchmarks()