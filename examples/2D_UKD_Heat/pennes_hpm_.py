import sys
from argparse import ArgumentParser
from datasets import InitialConditionDataset, PDEDataset, derivatives
from torch.autograd import Variable
from torch import Tensor, load, device

sys.path.append('../..')
import PINNFramework as pf

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--identifier", type=str, default="UKD_DeepHPM")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--epochs_pt", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--t_step", type=int, default=1)
    parser.add_argument("--pix_step", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_batches", type=int, default=28125)
    parser.add_argument("--hidden_size", type=int, default=500)
    parser.add_argument("--num_hidden", type=int, default=8)
    parser.add_argument("--hidden_size_hs", type=int, default=10)
    parser.add_argument("--num_hidden_hs", type=int, default=1)
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--use_horovod", type=int, default=1)
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--weight", type=float, default=1.)
    
    parser.add_argument("--name", type=str)
    parser.add_argument("--path_data", type=str)
    parser.add_argument("--num_t", type=int)
    parser.add_argument("--weight_j", type=float, default=1.)
    parser.add_argument("--pretrained", type=int, default=0)
    parser.add_argument("--pretrained_name", type=str, default='')
    
    parser.add_argument("--convection", type=int, default=1)
    parser.add_argument("--perfusion", type=int, default=1)
    parser.add_argument("--metabolism", type=int, default=1)
    parser.add_argument("--heat_source", type=int, default=1)
    parser.add_argument("--respiration", type=int, default=1)
    parser.add_argument("--heart_rate", type=int, default=1)
    parser.add_argument("--cooling", type=int, default=1)
    parser.add_argument("--cold_bolus", type=int, default=0)
    args = parser.parse_args()
    
    data_info = {
        "path_data": args.path_data,
        "num_t": args.num_t,
        "t_step": args.t_step,
        "pix_step": args.pix_step,
        "num_x": 640,
        "num_y": 480,
        "t_min": InitialConditionDataset.load_frame(args.path_data, 0)[1].item(),
        "t_max": InitialConditionDataset.load_frame(args.path_data, args.num_t)[1].item(),
        "spat_res": 0.3
    }

    segm_params = [32.4, 5, 5] if not args.cold_bolus else [31.0, 15, 15]
    # Initial condition
    num_batches = int(((args.num_t*640*480*0.25)*0.5)//args.batch_size)
    
    ic_dataset = InitialConditionDataset(
        data_info, args.batch_size, num_batches, segm_params)
  
    initial_condition = pf.InitialCondition(dataset=ic_dataset, name="Initial Condition", weight=args.weight, weight_j=args.weight_j)
    # PDE dataset
    pde_dataset = PDEDataset(
        data_info,
        args.batch_size,
        num_batches,
        segm_params)
            
    low_bound = Tensor([49.8000, 22.2000,  0.0]) if not args.cold_bolus else Tensor([34.2000, 4.5000,  0.0])
    up_bound = Tensor([148.5000, 120.6000, 60.0]) if not args.cold_bolus else Tensor([164.4000, 112.2000,  50.0])
      
    heat_source_net = pf.models.MLP(input_size=3,
                                output_size=1,
                                hidden_size=args.hidden_size_hs,
                                num_hidden=args.num_hidden_hs,
                                lb=low_bound[:3],
                                ub=up_bound[:3])
   
    # PINN model
    # Input: spatiotemporal coordinates of a point x,y,t
    # Output: temperature u at the point
    model = pf.models.MLP(input_size=3,
                          output_size=1,
                          hidden_size=args.hidden_size,
                          num_hidden=args.num_hidden,
                          lb=low_bound,
                          ub=up_bound)
    
    if args.pretrained:
        if len(args.pretrained_name):
            model.load_state_dict(load('./models/pretrained/' + args.pretrained_name + '.pt', map_location=device('cpu')))
        else:
            raise ValueError('pretrained model is not given')
   
    # HPM model: du/dt = k_values*(u_xx + u_yy) + heat_source
    # Initialization: k_values: Variable, heat source model: nn.Module
    # Forward pass input: output of the derivatives function for a point x,y,t
    # Forward pass output: du/dt value for the point
    config = {'convection': args.convection, 'perfusion': args.perfusion, 'metabolism': args.metabolism, 'heat_source': args.heat_source, 'respiration': args.respiration, 'heart_rate': args.heart_rate, 'cooling': args.cooling, 'cold_bolus': args.cold_bolus}
        
    hpm_model = pf.models.PennesHPM_(config, heat_source_net)
    hpm_loss = pf.HPMLoss.HPMLoss(dataset=pde_dataset, hpm_input=derivatives, hpm_model=hpm_model, name="Pennes Equation", weight=args.weight)
    
    pinn = pf.PINN(
        model,
        input_dimension=8,
        output_dimension=1,
        pde_loss=hpm_loss,
        initial_condition=initial_condition,
        boundary_condition=None,
        use_gpu=args.use_gpu,
        use_horovod=args.use_horovod)

    pinn.fit(args=args, epochs=args.epochs, epochs_pt=args.epochs_pt, optimizer='Adam', learning_rate=args.learning_rate, lbfgs_finetuning=False, pinn_path='./models/' + args.name+'_best_model_pinn.pt', hpm_path='./models/' + args.name+'_best_model_hpm.pt')
