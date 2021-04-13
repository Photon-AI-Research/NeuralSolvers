from torch.utils.data import DataLoader
from IC_Dataset import IC_Dataset as Dataset
from argparse import ArgumentParser
import horovod.torch as hvd
import sys
import torch.optim as optim
import numpy as np
import torch
sys.path.append('../..')  # PINNFramework etc.
import PINNFramework as pf
import wandb
from siren_pytorch import SirenNet
import matplotlib.pyplot as plt


def visualize_gt_diagnostics(dataset):
    e_field = dataset.e_field.reshape(256,2048,256)
    mean = np.mean(e_field)
    std = np.std(e_field)
    median = np.median(e_field)
    
    fig1 = plt.figure()
    slc = e_field[:,:,120]
    plt.imshow(slc,cmap='jet',aspect='auto')
    plt.colorbar()
    plt.xlabel("y")
    plt.ylabel("z")
    
    fig2 = plt.figure()
    slc = e_field[:,800,:]
    plt.imshow(slc,cmap='jet',aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("z")
    
    fig3 = plt.figure()
    slc = e_field[120,:,:]
    plt.imshow(slc,cmap='jet',aspect='auto')
    plt.colorbar()
    plt.xlabel("y")
    plt.ylabel("x")
    
    wandb.log(
        {
            "GT Laser xy":fig1,
            "GT Laser xz":fig2,
            "GT Laser yz":fig3,
            "GT Mean Prediction":mean,
            "GT Std Prediction":std,
            "GT Median Prediction": median  
            }
    )
    plt.close('all')

    

def diagnostics(model, dataset, eval_bs=1048576):
    with torch.no_grad():
        num_batches = int(dataset.input_x.shape[0] / eval_bs)
        outputs = []
        for idx in range(num_batches):
            x = torch.tensor(dataset.input_x[eval_bs * idx : eval_bs * (idx +1),:]).float().cuda()
            output = model(x).detach().cpu().numpy()
            outputs.append(output)
        pred = np.concatenate(outputs,axis=0)
        pred = pred.reshape(256,2048,256)
        mean= np.mean(pred) 
        std = np.std(pred)
        median = np.median(pred)
        fig1 = plt.figure()
        slc = pred[:,:,120]
        plt.imshow(slc,cmap='jet',aspect='auto')
        plt.colorbar()
        plt.xlabel("y")
        plt.ylabel("z")

        fig2 = plt.figure()
        slc = pred[:,800,:]
        plt.imshow(slc,cmap='jet',aspect='auto')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("z")

        fig3 = plt.figure()
        slc = pred[120,:,:]
        plt.imshow(slc,cmap='jet',aspect='auto')
        plt.colorbar()
        plt.xlabel("y")
        plt.ylabel("x")
        wandb.log(
            {
                "Laser xy":fig1,
                "Laser xz":fig2,
                "Laser yz":fig3,
                "Mean Prediction":mean,
                "Std Prediction":std,
                "Median Prediction": median
                
            }
        )

if __name__ == "__main__":
    
    hvd.init() 
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    print("Rank {} is active".format(rank))
    parser = ArgumentParser()
    parser.add_argument("--path", dest="path", type=str)
    parser.add_argument("--iteration", dest="iteration", type=int, default=0)
    parser.add_argument("--n0", dest="n0", type=int, default=int(130e6))
    parser.add_argument("--batch_size", dest="batch_size", type=int)
    parser.add_argument("--normalize_labels",dest="normalize_labels",type=int)
    parser.add_argument("--num_experts", dest="num_experts", type=int)
    parser.add_argument("--hidden_size", dest="hidden_size", type=int)
    parser.add_argument("--num_hidden", dest="num_hidden", type=int)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float)
    parser.add_argument("--model", dest="model", type=str,default="gpinn")
    parser.add_argument("--balance", dest="balance",type=float, default=1e-2)
    parser.add_argument("--frequency",dest="frequency",type=float, default=30.)
    parser.add_argument("--activation", dest="activation",type=str, default='tanh')
    parser.add_argument("--shuffle", dest="shuffle",type=int, default=0)
    parser.add_argument("--k",dest="k", type=int, default=1)
    args = parser.parse_args()
    if rank == 0:
        wandb.init(project='wave_equation_pinn', entity='aipp')
        wandb.config.update(args)
    

    dataset = Dataset(path=args.path,
                      iteration=args.iteration,
                      n0=args.n0,
                      batch_size=args.batch_size,
                      normalize_labels=args.normalize_labels)
    if rank == 0:
        visualize_gt_diagnostics(dataset)
    
    if args.activation =='tanh':
        activation = torch.tanh
    elif args.activation == 'sin':
        activation = torch.sin
        
    if args.model=="gpinn":
        model = pf.models.distMoe(input_size=4, output_size=1,
                                  num_experts=args.num_experts, hidden_size=args.hidden_size, num_hidden=args.num_hidden,
                                  lb=dataset.lb, ub=dataset.ub,activation=activation,k=args.k)
    if args.model=="siren":
        model = SirenNet(
            dim_in = 4,                     # input dimension, ex. 2d coor
            dim_hidden = args.hidden_size,  # hidden dimension
            dim_out = 1,                    # output dimension, ex. rgb value
            num_layers = args.num_hidden,   # number of layers
            w0_initial = args.frequency).cuda()    # different signals may require different omega_0 in the first layer - this is a hyperparameter
 
        
    if args.model =="mlp":
        model = pf.models.MLP(
            input_size = 4, 
            output_size = 1,
            hidden_size = args.hidden_size,
            num_hidden = args.num_hidden, 
            lb=dataset.lb,
            ub=dataset.ub,
            activation=activation)
        model.cuda()
    
    if args.model =="finger":
        model = pf.models.FingerNet(lb=dataset.lb,ub=dataset.ub,activation=torch.sin)
        model.cuda()
            
    if args.model == "snake":
        model = pf.models.SnakeMLP(input_size=4,
                                   output_size=1,
                                   lb=dataset.lb, 
                                   ub=dataset.ub,
                                   frequency=args.frequency,
                                   hidden_size=args.hidden_size,
                                   num_hidden=8)
        model.cuda()

    if rank==0:
        wandb.watch(model,log='all')

    initial_condition = pf.InitialCondition(dataset)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # Add Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset, num_replicas=hvd.size(), rank=hvd.rank())
    loader = DataLoader(dataset, shuffle=args.shuffle, sampler=train_sampler)
    best_epoch_loss = np.inf
    for epoch in range(1, args.num_epochs+1):
        mean_epoch_loss = 0
        for idx, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            x = x[0]
            y = y[0]
            x = x.cuda()
            y = y.cuda()
            if args.model=="siren":
                lb = torch.tensor(dataset.lb).float().cuda()
                ub = torch.tensor(dataset.ub).float().cuda()
                x = 2.0*(x - lb)/(ub - lb) - 1.0
                ic_loss = initial_condition(x, model, y)
            elif args.model=="gpinn":
                ic_loss = initial_condition(x, model, y) + model.loss
            else:
                ic_loss = initial_condition(x, model, y)
            ic_loss.backward()
            optimizer.step()
            mean_epoch_loss += ic_loss / len(dataset)  # length of dataset is equal to amount of batches
        if rank == 0:
            wandb.log({'initial_condition': mean_epoch_loss})
            if args.model == "gpinn":
                wandb.log({'balance_loss': model.loss})
            print("Epoch: {} Loss: {}".format(epoch, mean_epoch_loss))
            if mean_epoch_loss < best_epoch_loss:
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'loss': mean_epoch_loss}, 'checkpoints/{}.pt'.format(wandb.run.name))
                best_epoch_loss = mean_epoch_loss
                diagnostics(model,dataset)




