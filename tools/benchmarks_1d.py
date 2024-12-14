from examples.Heat_Equation_1d import Heat_Equation
from examples.Schroedinger_1d import Schroedinger
from examples.Burgers_Equation_1d import Burgers_Equation
from NeuralSolvers.loggers.Python_Logger import PythonLogger
from NeuralSolvers.models import ModulatedMLP
import numpy as np
import torch
import scipy

DEVICE = 'cuda'
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

    print("*** Burgers Equation ***")
    ## Load data for modulation network
    data = scipy.io.loadmat('../examples/Burgers_Equation_1d/burgers_shock.mat')
    u_i = torch.Tensor(np.real(data['usol']).T).float().to('cuda')

    ## Modulated PINN
    model = ModulatedMLP(
        input_size=2, output_size=1, device=DEVICE,
        hidden_size=40, num_hidden=8, lb=DOMAIN_LOWER_BOUND, ub=DOMAIN_UPPER_BOUND,
        activation=torch.tanh, u_i=u_i
    )
    pinn = Burgers_Equation.setup_pinn(model=model, file_path = '../examples/Burgers_Equation_1d/burgers_shock.mat')

    #vitl16 with sigmoid
    #[2024-12-14 22:20:30.113214]:Epoch 1000/1000 | PINN Loss 0.0093754334 | Initial Condition loss: 0.005757 | PDE loss: 0.003618 | Epoch Duration 0.03798

    ## original PINN
    # [2024-12-14 21:57:03.422603]:Epoch 1000/1000 | PINN Loss 0.0302316695 | Initial Condition loss: 0.020797 | PDE loss: 0.009435 | Epoch Duration 0.00928

    # train
    Burgers_Equation.train_pinn(pinn, Burgers_Equation.NUM_EPOCHS, logger=burger_log)
    if(plot_results_debug):
        t, x, exact_solution = Burgers_Equation.load_burger_data(file_path = '../examples/Burgers_Equation_1d/burgers_shock.mat')
        Burgers_Equation.plot_solution(pinn, t, x, exact_solution)
    return
    print("*** Heat Equation ***")
    pinn = Heat_Equation.setup_pinn()
    Heat_Equation.train_pinn(pinn, Heat_Equation.NUM_EPOCHS, logger=heat_log)
    if (plot_results_debug):
        Heat_Equation.plot_solution(pinn)
        Heat_Equation.plot_analytical_solution()

    print("*** Schroedinger Equation ***")
    file_path = '../examples/Schroedinger_1d/NLS.mat'
    pinn = Schroedinger.setup_pinn(file_path=file_path)
    Schroedinger.train_pinn(pinn, Schroedinger.NUM_EPOCHS, logger=schrodinger_log)
    if (plot_results_debug):
        Schroedinger.plot_solution(pinn, file_path=file_path)
        Schroedinger.plot_exact_solution(file_path=file_path)
        Schroedinger.compare_solutions(pinn, file_path=file_path)

    print("*** Burgers Equation ***")
    print(", ".join([f"{key}: {value}" for key, value in burger_log.loss_history.items()]))
    print("*** Heat Equation ***")
    print(", ".join([f"{key}: {value}" for key, value in heat_log.loss_history.items()]))
    print("*** Schroedinger Equation ***")
    print(", ".join([f"{key}: {value}" for key, value in schrodinger_log.loss_history.items()]))



if __name__ == "__main__":
    run_benchmarks()