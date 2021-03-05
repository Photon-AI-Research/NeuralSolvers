from torch.utils.data import DataLoader
from IC_Dataset import IC_Dataset as Dataset
from argparse import ArgumentParser

import sys
import torch.optim as optim
import numpy as np
import torch
sys.path.append('../..')  # PINNFramework etc.
import PINNFramework as pf
import wandb

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", dest="path", type=str)
    parser.add_argument("--iteration", dest="iteration", type=int, default=0)
    parser.add_argument("--n0", dest="n0", type=int, default=int(31e6))
    parser.add_argument("--batch_size", dest="batch_size", type=int)
    parser.add_argument("--num_experts", dest="num_experts", type=int)
    parser.add_argument("--hidden_size", dest="hidden_size", type=int)
    parser.add_argument("--num_hidden", dest="num_hidden", type=int)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float)
    args = parser.parse_args()
    wandb.init(project='wave_equation_pinn', entity='aipp')
    wandb.config.update(args)
    dataset = Dataset(path=args.path, iteration=args.iteration, n0=args.n0, batch_size=args.batch_size)
    model = pf.models.distMoe(input_size=4, output_size=3,
                              num_experts=args.num_experts, hidden_size=args.hidden_size, num_hidden=args.num_hidden,
                              lb=dataset.lb, ub=dataset.ub, device='cuda:0')

    wandb.watch(model)

    initial_condition = pf.InitialCondition(dataset)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(dataset, num_workers=16, pin_memory=True)
    best_epoch_loss = np.inf
    for epoch in range(1, args.num_epochs+1):
        mean_epoch_loss = 0
        for idx, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            ic_loss = initial_condition(x, model, y) + model.loss
            ic_loss.backward()
            optimizer.step()
            mean_epoch_loss += ic_loss / len(dataset)  # length of dataset is equal to amount of batches
        wandb.log({'initial_condition': mean_epoch_loss})
        print("Epoch: {} Loss: {}".format(epoch, mean_epoch_loss))
        if mean_epoch_loss < best_epoch_loss:
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': mean_epoch_loss}, '{}.pt'.format(wandb.run.name))




