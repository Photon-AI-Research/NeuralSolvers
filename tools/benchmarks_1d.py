from examples.Heat_Equation_1d import Heat_Equation
from examples.Schroedinger_1d import Schroedinger
from examples.Burgers_Equation_1d import Burgers_Equation


def run_benchmarks():

    print("*** Burgers Equation ***")
    pinn = Burgers_Equation.setup_pinn(file_path = '../examples/Burgers_Equation_1d/burgers_shock.mat')
    Burgers_Equation.train_pinn(pinn, Burgers_Equation.NUM_EPOCHS)
    print(pinn.loss_log)
    t, x, exact_solution = Burgers_Equation.load_burger_data(file_path = '../examples/Burgers_Equation_1d/burgers_shock.mat')
    Burgers_Equation.plot_solution(pinn, t, x, exact_solution)


    print("*** Heat Equation ***")
    pinn = Heat_Equation.setup_pinn()
    Heat_Equation.train_pinn(pinn, Heat_Equation.NUM_EPOCHS)
    Heat_Equation.plot_solution(pinn)
    Heat_Equation.plot_analytical_solution()

    print("*** Schroedinger Equation ***")
    file_path = '../examples/Schroedinger_1d/NLS.mat'
    pinn = Schroedinger.setup_pinn(file_path=file_path)
    Schroedinger.train_pinn(pinn, Schroedinger.NUM_EPOCHS)
    Schroedinger.plot_solution(pinn, file_path=file_path)
    Schroedinger.plot_exact_solution(file_path=file_path)
    Schroedinger.compare_solutions(pinn, file_path=file_path)

if __name__ == "__main__":
    run_benchmarks()