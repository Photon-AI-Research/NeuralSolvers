import sys
from Datasets import *
from argparse import ArgumentParser
sys.path.append('../..')  # PINNFramework etc.
import PINNFramework as pf

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--identifier", type=str, default="UKD_DeepHPM")
    parser.add_argument("--pData", type=str, default="/home/maxxxzdn/DeepHPM/ThermalImaging/data/")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--nt", type=int, default=1000)
    parser.add_argument("--timeStep", type=int, default=25)
    parser.add_argument("--batchSize", type=int, default=512)
    parser.add_argument("--numBatches", type=int, default=2e5)
    parser.add_argument("--hidden_size", type=int, default=500)
    parser.add_argument("--num_hidden", type=int, default=8)
    parser.add_argument("--hidden_size_alpha", type=int, default=500)
    parser.add_argument("--num_hidden_alpha", type=int, default=8)
    parser.add_argument("--hidden_size_hs", type=int, default=500)
    parser.add_argument("--num_hidden_hs", type=int, default=8)
    args = parser.parse_args()

    # Initial condition
    ic_dataset = InitialConditionDataset(
        pData=args.pData,
        batchSize=args.batchSize,
        numBatches=args.numBatches,
        nt=args.nt,
        timeStep=args.timeStep)
    initial_condition = pf.InitialCondition(ic_dataset)
    # PDE dataset
    pde_dataset = PDEDataset(
        pData=args.pData,
        seg_mask=ic_dataset.seg_mask,
        batchSize=args.batchSize,
        numBatches=args.numBatches,
        nt=args.nt,
        timeStep=args.timeStep,
        t_ub=ic_dataset.cSystem["t_ub"])

    # Thermal diffusivity model
    # Input: spatiotemporal coordinates of a point x,y,t
    # Output: thermal diffusivity value for the point
    alpha_net = pf.models.MLP(input_size=3,
                              output_size=1,
                              hidden_size=args.hidden_size_alpha,
                              num_hidden=args.num_hidden_alpha,
                              lb=ic_dataset.lb[:3], #lb for x,y,t
                              ub=ic_dataset.ub[:3]) #ub for x,y,t
    # Heat source model - part of du/dt that cannot be explained by conduction
    # Input: spatiotemporal coordinates of a point x,y,t
    # Output: heat source value for the point
    heat_source_net = pf.models.MLP(input_size=3,
                                    output_size=1,
                                    hidden_size=args.hidden_size_hs,
                                    num_hidden=args.num_hidden_hs,
                                    lb=ic_dataset.lb[:3], #lb for x,y,t
                                    ub=ic_dataset.ub[:3]) #ub for x,y,t
    # PINN model
    # Input: spatiotemporal coordinates of a point x,y,t
    # Output: temperature u at the point
    model = pf.models.MLP(input_size=3,
                          output_size=1,
                          hidden_size=args.hidden_size,
                          num_hidden=args.num_hidden,
                          lb=ic_dataset.lb[:3], #lb for x,y,t
                          ub=ic_dataset.ub[:3]) #ub for x,y,t
    # HPM model: du/dt = alpha*(u_xx + u_yy) + heat_source
    # Initialization: alpha model, heat source model
    # Forward pass input: output of the derivatives function for a point x,y,t
    # Forward pass output: du/dt value for the point
    hpm_model = pf.models.MultiModelHPM(alpha_net, heat_source_net)
    hpm_loss = pf.HPMLoss.HPMLoss(pde_dataset, derivatives, hpm_model)
    pinn = pf.PINN(
        model,
        input_dimension=6,
        output_dimension=1,
        pde_loss=hpm_loss,
        initial_condition=initial_condition,
        boundary_condition=None,
        use_gpu=False)

    pinn.fit(args.epochs, 'Adam', 1e-6)
