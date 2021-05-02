import sys
from argparse import ArgumentParser
from datasets import InitialConditionDataset, PDEDataset, derivatives
from torch.autograd import Variable
from torch import Tensor

sys.path.append('../..')
import PINNFramework as pf

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="ukd")
    parser.add_argument("--identifier", type=str, default="UKD_DeepHPM")
    parser.add_argument("--path_data", type=str, default="./data/")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_t", type=int, default=1000)
    parser.add_argument("--t_step", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_batches", type=int, default=2e5)
    parser.add_argument("--hidden_size", type=int, default=500)
    parser.add_argument("--num_hidden", type=int, default=8)
    parser.add_argument("--hidden_size_hs", type=int, default=200)
    parser.add_argument("--num_hidden_hs", type=int, default=4)
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--use_horovod", type=int, default=0)
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--regularize", type=int, default=0)
    parser.add_argument("--weight", type=float, default=1.)
    parser.add_argument("--weight_j", type=float, default=1.)
    parser.add_argument("--erode", type=int, default=0)
    parser.add_argument("--delay", type=int, default=0)
    args = parser.parse_args()
    
    data_info = {
        "path_data": args.path_data,
        "num_t": args.num_t,
        "t_step": args.t_step,
        "pix_step": 1,
        "num_x": 640,
        "num_y": 480,
        "t_min": InitialConditionDataset.load_frame(args.path_data, 0)[1].item(),
        "t_max": InitialConditionDataset.load_frame(args.path_data, args.num_t)[1].item(),
        "spat_res": 0.3
    }

    # Initial condition
    ic_dataset = InitialConditionDataset(
        data_info, args.batch_size, args.num_batches)
    if args.use_gpu:
        ic_dataset.cuda()        
    initial_condition = pf.InitialCondition(ic_dataset)
    # PDE dataset
    pde_dataset = PDEDataset(
        data_info,
        args.batch_size,
        args.num_batches,
        args.erode)
    if args.use_gpu:
        pde_dataset.cuda()     
    #Thermal conductivity values on image grid
    #k_value = 0.12 #Variable(Tensor([0.1]))
    #p_value = -0.09 #Variable(Tensor([-0.09]))
    #heat_source = Variable(Tensor([0]))
    
    heat_source_net = pf.models.MLP(input_size=2,
                                output_size=1,
                                hidden_size=args.hidden_size_hs,
                                num_hidden=args.num_hidden_hs,
                                lb=ic_dataset.low_bound[:2],
                                ub=ic_dataset.up_bound[:2])
   
    # PINN model
    # Input: spatiotemporal coordinates of a point x,y,t
    # Output: temperature u at the point
    model = pf.models.MLP(input_size=3,
                          output_size=1,
                          hidden_size=args.hidden_size,
                          num_hidden=args.num_hidden,
                          lb=ic_dataset.low_bound,
                          ub=ic_dataset.up_bound)
   
    # HPM model: du/dt = k_values*(u_xx + u_yy) + heat_source
    # Initialization: k_values: Variable, heat source model: nn.Module
    # Forward pass input: output of the derivatives function for a point x,y,t
    # Forward pass output: du/dt value for the point
        
    heat_source_net.cuda()
    hpm_model = pf.models.PennesHPM(heat_source_net)
    if args.use_gpu:
        hpm_model.cuda()
    hpm_loss = pf.HPMLoss.HPMLoss(pde_dataset, derivatives, hpm_model, weight=args.weight)
    
    pinn = pf.PINN(
        model,
        input_dimension=8,
        output_dimension=1,
        pde_loss=hpm_loss,
        initial_condition=initial_condition,
        boundary_condition=None,
        use_gpu=args.use_gpu,
        use_horovod=args.use_horovod,
        use_wandb=args.use_wandb,
        project_name='thermal_hpm',
        regularize=args.regularize,
        weight_j=args.weight_j,
        delay = args.delay)

    pinn.fit(args.epochs, 'Adam', args.learning_rate, pinn_path=args.name+'_best_model_pinn.pt', hpm_path=args.name+'_best_model_hpm.pt')