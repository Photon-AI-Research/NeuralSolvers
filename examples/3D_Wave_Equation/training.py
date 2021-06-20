from torch.utils.data import DataLoader
from IC_Dataset import ICDataset as ICDataset
from PDE_Dataset import PDEDataset as PDEDataset
from BC_Dataset import BoundaryDataset as BCDataset
from argparse import ArgumentParser
import sys

import numpy as np
import torch

sys.path.append('../..')  # PINNFramework etc.
import PINNFramework as pf
import wandb
import matplotlib.pyplot as plt
from torch.autograd import grad


def visualize_gt_diagnostics(dataset, time_step):
    e_field = dataset.e_field.reshape(256, 2048, 256)

    fig1 = plt.figure()
    slc = e_field[:, :, 120]
    plt.imshow(slc, cmap='jet', aspect='auto')
    plt.colorbar()
    plt.xlabel("y")
    plt.ylabel("z")

    fig2 = plt.figure()
    slc = e_field[:, 800, :]
    plt.imshow(slc, cmap='jet', aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("z")

    fig3 = plt.figure()
    slc = e_field[120, :, :]
    plt.imshow(slc, cmap='jet', aspect='auto')
    plt.colorbar()
    plt.xlabel("y")
    plt.ylabel("x")

    wandb.log(
        {
            "GT Laser xy Time: {}".format(time_step): fig1,
            "GT Laser xz Time: {}".format(time_step): fig2,
            "GT Laser yz time: {}".format(time_step): fig3,

        }
    )
    plt.close('all')


class VisualisationCallback(pf.callbacks.Callback):
    def __init__(self, model, logger, time_step, eval_bs=1048576):
        self.model = model
        self.logger = logger
        self.eval_bs = eval_bs
        self.time_step = time_step
        # gives me grid with time at given time point
        self.dataset = ICDataset(args.path, time_step, 0, 0, args.max_t, args.normalize_labels)

    def __call__(self, epoch):
        with torch.no_grad():
            num_batches = int(self.dataset.input_x.shape[0] / self.eval_bs)
            outputs = []
            for idx in range(num_batches):
                x = torch.tensor(self.dataset.input_x[self.eval_bs * idx: self.eval_bs * (idx + 1), :]).float().cuda()
                output = model(x).detach().cpu().numpy()
                outputs.append(output)

            pred = np.concatenate(outputs, axis=0)
            pred = pred.reshape(256, 2048, 256)

            fig1 = plt.figure()
            slc = pred[:, :, 120]
            plt.imshow(slc, cmap='jet', aspect='auto')
            plt.colorbar()
            plt.xlabel("y")
            plt.ylabel("z")

            fig2 = plt.figure()
            slc = pred[:, 800, :]
            plt.imshow(slc, cmap='jet', aspect='auto')
            plt.colorbar()
            plt.xlabel("x")
            plt.ylabel("z")

            fig3 = plt.figure()
            slc = pred[120, :, :]
            plt.imshow(slc, cmap='jet', aspect='auto')
            plt.colorbar()
            plt.xlabel("y")
            plt.ylabel("x")
            
            logger.log_image(fig1, "YZ Time: {}".format(self.time_step), epoch)
            logger.log_image(fig2, "XZ Time: {}".format(self.time_step), epoch)
            logger.log_image(fig3, "YX Time: {}".format(self.time_step), epoch)
            plt.close('all')



def wave_eq(x, u):
    grads = torch.ones(u.shape, device=u.device)  # move to the same device as prediction
    # calculate first order derivatives
    grad_u = grad(u, x, create_graph=True, grad_outputs=grads)[0]  # (z, y, x, t)

    u_z = grad_u[:, 0]
    u_y = grad_u[:, 1]
    u_x = grad_u[:, 2]
    u_t = grad_u[:, 3]

    grads = torch.ones(u_z.shape, device=u_z.device) # update for shapes
    # calculate second order derivatives
    u_zz = grad(u_z, x, create_graph=True, grad_outputs=grads)[0][:, 0]  # (z, y, x, t)
    u_yy = grad(u_y, x, create_graph=True, grad_outputs=grads)[0][:, 1]
    u_xx = grad(u_x, x, create_graph=True, grad_outputs=grads)[0][:, 2]
    u_tt = grad(u_t, x, create_graph=True, grad_outputs=grads)[0][:, 3]
    f_u = u_tt - (u_zz + u_yy + u_xx)
    return f_u

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = ArgumentParser()
    parser.add_argument("--name", dest="name", type=str)
    parser.add_argument("--path", dest="path", type=str)
    parser.add_argument("--iteration", dest="iteration", type=int, default=2000)
    parser.add_argument("--n0", dest="n0", type=int, default=int(134e6))
    parser.add_argument("--nf", dest="nf", type=int, default=int(130e9))
    parser.add_argument("--nb", dest="nb", type=int, default=int(5e6))
    parser.add_argument("--batch_size_n0", dest="batch_size_n0", type=int, default=50000)
    parser.add_argument("--batch_size_nf", dest="batch_size_nf", type=int, default=50000)
    parser.add_argument("--batch_size_nb", dest="batch_size_nb", type=int, default=50000)
    parser.add_argument("--normalize_labels", dest="normalize_labels", type=int)
    parser.add_argument("--num_experts", dest="num_experts", type=int)
    parser.add_argument("--hidden_size", dest="hidden_size", type=int)
    parser.add_argument("--num_hidden", dest="num_hidden", type=int)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float)
    parser.add_argument("--model", dest="model", type=str, default="gpinn")
    parser.add_argument("--balance", dest="balance", type=float, default=1e-2)
    parser.add_argument("--frequency", dest="frequency", type=float, default=30.)
    parser.add_argument("--activation", dest="activation", type=str, default='tanh')
    parser.add_argument("--shuffle", dest="shuffle", type=int, default=0)
    parser.add_argument("--annealing",dest="annealing",type=int, default=0)
    parser.add_argument("--k", dest="k", type=int, default=1)
    parser.add_argument("--max_t",dest="max_t", type=int, default=3000)
    args = parser.parse_args()
    ic_dataset = ICDataset(path=args.path,
                           iteration=args.iteration,
                           n0=args.n0,
                           batch_size=args.batch_size_n0,
                           max_t=args.max_t,
                           normalize_labels=args.normalize_labels)
    print("ic",len(ic_dataset))
    initial_condition = pf.InitialCondition(ic_dataset, "Initial Condition")
    pde_dataset = PDEDataset(ic_dataset.lb, ic_dataset.ub, args.nf, args.batch_size_nf, iterative_generation=True)
    print("pde",len(pde_dataset))
    pde_condition = pf.PDELoss(pde_dataset, wave_eq, "Wave Equation")
    boundary_dataset = BCDataset(ic_dataset.lb, ic_dataset.ub, args.nb, args.batch_size_nb, period=1)
    boundary_condition = pf.PeriodicBC(boundary_dataset, 0, "Periodic Boundary Condition")
    if args.activation == 'tanh':
        activation = torch.tanh
    elif args.activation == 'sin':
        activation = torch.sin

    if args.model == "gpinn":
        model = pf.models.distMoe(input_size=4, output_size=1,
                                  num_experts=args.num_experts, hidden_size=args.hidden_size,
                                  num_hidden=args.num_hidden,
                                  lb=ic_dataset.lb, ub=ic_dataset.ub, activation=activation, k=args.k)

    if args.model == "mlp":
        model = pf.models.MLP(
            input_size=4,
            output_size=1,
            hidden_size=args.hidden_size,
            num_hidden=args.num_hidden,
            lb=ic_dataset.lb,
            ub=ic_dataset.ub,
            activation=activation)
        model.cuda()

    if args.model == "finger":
        model = pf.models.FingerNet(numFeatures=300,lb=ic_dataset.lb, ub=ic_dataset.ub, activation=torch.sin)
        model.cuda()

    if args.model == "snake":
        model = pf.models.SnakeMLP(input_size=4,
                                   output_size=1,
                                   lb=ic_dataset.lb,
                                   ub=ic_dataset.ub,
                                   frequency=args.frequency,
                                   hidden_size=args.hidden_size,
                                   num_hidden=8)
        model.cuda()


    pinn = pf.PINN(model=model,
                   input_dimension=4,
                   output_dimension=1,
                   pde_loss=pde_condition,
                   initial_condition=initial_condition,
                   boundary_condition=[],
                   use_gpu=True,
                   use_horovod=True,
                   dataset_mode='max'
                   )
    if pinn.rank == 0:
        logger = pf.WandbLogger(project='wave_equation_pinn', args=args, entity='aipp')
        wandb.watch(model, log='all')
        # visualization callbacks
        cb_2000 = VisualisationCallback(model, logger, 2000)
        cb_2100 = VisualisationCallback(model, logger, 2100)
        cb_list = pf.callbacks.CallbackList([cb_2000, cb_2100])

    else:
        logger = None
        cb_list = None
    checkpoint_path = "checkpoints/" + args.name + "_checkpoint.pt"
    print("callbacks are finished") 
    #write ground truth diagnostics
    if pinn.rank == 0:
        visualize_gt_diagnostics(cb_2000.dataset, 2000)
        visualize_gt_diagnostics(cb_2100.dataset, 2100)
    print("start fit")
    pinn.fit(epochs=args.num_epochs,
             optimizer='Adam',
             learning_rate=args.learning_rate,
             pretraining=True,
             epochs_pt=30,
             lbfgs_finetuning=False,
             writing_cylcle=5,
             activate_annealing=args.annealing,
             logger=logger,
             checkpoint_path=checkpoint_path,
             restart=True,
             callbacks=cb_list,
             pinn_path="best_model_" + args.name + '.pt'
             )
