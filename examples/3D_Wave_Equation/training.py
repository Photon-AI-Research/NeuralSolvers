from torch.utils.data import DataLoader
from IC_Dataset import ICDataset as ICDataset
from PDE_Dataset import PDEDataset as PDEDataset
from argparse import ArgumentParser
import sys
import torch.optim as optim
import numpy as np
import torch

sys.path.append('../..')  # PINNFramework etc.
import PINNFramework as pf
import wandb
from siren_pytorch import SirenNet
import matplotlib.pyplot as plt
from torch.autograd import grad


def visualize_gt_diagnostics(dataset):
    e_field = dataset.e_field.reshape(256, 2048, 256)
    mean = np.mean(e_field)
    std = np.std(e_field)
    median = np.median(e_field)

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
            "GT Laser xy": fig1,
            "GT Laser xz": fig2,
            "GT Laser yz": fig3,
            "GT Mean Prediction": mean,
            "GT Std Prediction": std,
            "GT Median Prediction": median
        }
    )
    plt.close('all')


def diagnostics(model, dataset, eval_bs=1048576):
    with torch.no_grad():
        num_batches = int(dataset.input_x.shape[0] / eval_bs)
        outputs = []
        for idx in range(num_batches):
            x = torch.tensor(dataset.input_x[eval_bs * idx: eval_bs * (idx + 1), :]).float().cuda()
            output = model(x).detach().cpu().numpy()
            outputs.append(output)
        pred = np.concatenate(outputs, axis=0)
        pred = pred.reshape(256, 2048, 256)
        mean = np.mean(pred)
        std = np.std(pred)
        median = np.median(pred)
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
        wandb.log(
            {
                "Laser xy": fig1,
                "Laser xz": fig2,
                "Laser yz": fig3,
                "Mean Prediction": mean,
                "Std Prediction": std,
                "Median Prediction": median

            }
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", dest="path", type=str)
    parser.add_argument("--iteration", dest="iteration", type=int, default=0)
    parser.add_argument("--n0", dest="n0", type=int, default=int(134e6))
    parser.add_argument("--nf", dest="nf", type=int, default=int(130e7))
    parser.add_argument("--batch_size_n0", dest="batch_size_n0", type=int, default=50000)
    parser.add_argument("--batch_size_nf", dest="batch_size_nf", type=int, default=50000)
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
    args = parser.parse_args()

    wandb.init(project='wave_equation_pinn', entity='aipp')
    wandb.config.update(args)

    ic_dataset = ICDataset(path=args.path,
                           iteration=args.iteration,
                           n0=args.n0,
                           batch_size=args.batch_size_nf,
                           normalize_labels=args.normalize_labels)

    pde_dataset = PDEDataset(ic_dataset.lb, ic_dataset.ub, args.nf, args.batch_size_nf)
    visualize_gt_diagnostics(ic_dataset)

    if args.activation == 'tanh':
        activation = torch.tanh
    elif args.activation == 'sin':
        activation = torch.sin

    if args.model == "gpinn":
        model = pf.models.distMoe(input_size=4, output_size=1,
                                  num_experts=args.num_experts, hidden_size=args.hidden_size,
                                  num_hidden=args.num_hidden,
                                  lb=ic_dataset.lb, ub=ic_dataset.ub, activation=activation, k=args.k)
    if args.model == "siren":
        model = SirenNet(
            dim_in=4,  # input dimension, ex. 2d coor
            dim_hidden=args.hidden_size,  # hidden dimension
            dim_out=1,  # output dimension, ex. rgb value
            num_layers=args.num_hidden,  # number of layers
            w0_initial=args.frequency).cuda()

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
        model = pf.models.FingerNet(lb=ic_dataset.lb, ub=ic_dataset.ub, activation=torch.sin)
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

    def wave_eq(x, u):
        grads = torch.ones(u.shape, device=u.device)  # move to the same device as prediction
        # calculate first order derivatives
        grad_u = grad(u, x, create_graph=True, grad_outputs=grads)[0]  # (z, y, x, t)

        u_z = grad_u[:, 0]
        u_y = grad_u[:, 1]
        u_x = grad_u[:, 2]
        u_t = grad_u[:, 3]

        # calculate second order derivatives
        u_zz = grad(u_z, x, create_graph=True, grad_outputs=grads)[0][:, 0]  # (z, y, x, t)
        u_yy = grad(u_y, x, create_graph=True, grad_outputs=grads)[0][:, 1]
        u_xx = grad(u_x, x, create_graph=True, grad_outputs=grads)[0][:, 2]
        u_tt = grad(u_t, x, create_graph=True, grad_outputs=grads)[0][:, 3]

        f_u = u_tt - (u_zz + u_yy + u_xx)
        return f_u

    logger = pf.WandbLogger(project='wave_equation_pinn', entity='aipp')
    wandb.watch(model,log='all')
    initial_condition = pf.InitialCondition(ic_dataset)
    pde_condition = pf.PDELoss(pde_dataset, wave_eq)

    pinn = pf.PINN(model=model,
                   input_dimension=4,
                   output_dimension=1,
                   pde_loss=pde_condition,
                   initial_condition=initial_condition,
                   boundary_condition=None,
                   use_gpu=True,
                   use_horovod=True
                   )

    pinn.fit(epochs=args.num_epochs,
             optimizer='Adam',
             learning_rate=args.learning_rate,
             pretraining=True,
             epochs_pt=100,
             lbfgs_finetuning=False,
             activate_annealing=args.annealing
             )
