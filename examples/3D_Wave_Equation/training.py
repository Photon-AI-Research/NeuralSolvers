from torch.utils.data import DataLoader
from IC_Dataset import ICDataset as ICDataset
from PDE_Dataset import PDEDataset as PDEDataset
from TD_BC_Dataset import TDBCDataset as DTDataset
from BC_Dataset import BoundaryDataset as BCDataset
from argparse import ArgumentParser
import sys
from multiprocessing import Process, Value
import numpy as np
import torch
import horovod.torch as hvd
sys.path.append('../..')  # PINNFramework etc.
import PINNFramework as pf
import wandb
import matplotlib.pyplot as plt
from torch.autograd import grad
import pathlib
from pynvml import *
from pynvml.smi import nvidia_smi
import time


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
            del pred # clear memory
            logger.log_image(fig1, "YZ Time: {}".format(self.time_step), epoch)
            logger.log_image(fig2, "XZ Time: {}".format(self.time_step), epoch)
            logger.log_image(fig3, "YX Time: {}".format(self.time_step), epoch)
            plt.close('all')


class PerformanceCallback(pf.callbacks.Callback):
    def __init__(self, rank):
        self.lock = Value('i', 1)
        # start benchmark process
        print('benchmark')
        sys.stdout.flush()
        self.rank = rank
        if rank % 6 == 0:
            print('start benchmark')
            sys.stdout.flush()
            benchmark_process = Process(target=self.__call__, args=())
            benchmark_process.start()

    def __call__(self):
        nvmlInit()
        vars()[self.rank] = np.array([])
        timestamps = np.array([])
        start_time = time.time()
        deviceIdx = hvd.local_rank()  # GPU id
        nvsmi = nvidia_smi.getInstance()
        while self.lock.value != 0:
            time.sleep(10)
            res = nvsmi.DeviceQuery()
            vars()[hvd.rank()] = np.append(vars()[hvd.rank()], res['gpu'])
            timestamps = np.append(timestamps, time.time() - start_time)
        runtime = time.time() - start_time
        vars()[hvd.rank()][0]["runtime"] = runtime
        vars()[hvd.rank()][0]["timestamps"] = timestamps.shape
        vars()[hvd.rank()] = np.append(vars()[hvd.rank()], timestamps)
        np.save("/beegfs/global0/ws/s7520458-pinn_wave/Neuralexamples/3D_Wave_Equation/benchmarks/exp_{}_gpu_s2d_util_{}".format(1,
                                                                                                          hvd.rank()),
                vars()[hvd.rank()])
        print("done")
        sys.stdout.flush()
        return


def wave_eq(x, u):

    grads = torch.ones(u.shape, device=u.device)  # move to the same device as prediction

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
    #relu6 = torch.nn.ReLU6()
    #propagation_error = float(1./6.) * relu6(u_y*u_t)
    return f_u

if __name__ == "__main__":
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
    parser.add_argument("--annealing", dest="annealing",type=int, default=0)
    parser.add_argument("--k", dest="k", type=int, default=1)
    parser.add_argument("--boundary", dest="boundary", default=0)
    parser.add_argument("--max_t", dest="max_t", type=int, default=3000)
    parser.add_argument("--restart", dest="restart", type=int, default=1)
    parser.add_argument("--checkpoint", dest="checkpoint", type=str)
    args = parser.parse_args()
    ic_dataset = ICDataset(path=args.path,
                           iteration=args.iteration,
                           n0=args.n0,
                           batch_size=args.batch_size_n0,
                           max_t=args.max_t,
                           normalize_labels=args.normalize_labels)
    print("ic", len(ic_dataset))
    initial_condition = pf.InitialCondition(ic_dataset, "Initial Condition")
    pde_dataset = PDEDataset(args.path, args.nf, args.batch_size_nf, args.iteration, args.max_t)
    print("pde", len(pde_dataset))
    pde_condition = pf.PDELoss(pde_dataset, wave_eq, "Wave Equation")

    bc_dataset = DTDataset(args.path, args.iteration, args.nb, args.batch_size_nb)
    dt_boundary = pf.TimeDerivativeBC(bc_dataset, "Time Derivative Boundary")
    #boundary_x = pf.PeriodicBC(BCDataset(ic_dataset.lb, ic_dataset.ub, args.nb, args.batch_size_nb, 2), 0, "Boundary x")
    #boundary_y = pf.PeriodicBC(BCDataset(ic_dataset.lb, ic_dataset.ub, args.nb, args.batch_size_nb, 1), 0, "Boundary y")
    #boundary_z = pf.PeriodicBC(BCDataset(ic_dataset.lb, ic_dataset.ub, args.nb, args.batch_size_nb, 0), 0, "Boundary z")

    boundary_conditions = [dt_boundary]

    if args.activation == 'tanh':
        activation = torch.tanh
    elif args.activation == 'sin':
        activation = torch.sin

    if args.model == "gpinn":
        model = pf.models.FingerMoE(4,
                                    1,
                                    args.num_experts,
                                    args.hidden_size,
                                    args.num_hidden,
                                    ic_dataset.lb,
                                    ic_dataset.ub,
                                    torch.sin,
                                    scaling_factor=1.)
        model.cuda()

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
        model = pf.models.FingerNet(numFeatures=args.hidden_size,
                                    numLayers=args.num_hidden,
                                    lb=ic_dataset.lb,
                                    ub=ic_dataset.ub,
                                    activation=torch.sin,
                                    normalize=True,
                                    scaling=ic_dataset.e_field_max
                                    )
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
                   boundary_condition=boundary_conditions,
                   use_gpu=True,
                   use_horovod=True,
                   dataset_mode='max'
                   )
    if pinn.rank == 0:
        logger = pf.WandbLogger(project='wave_equation_pinn', args=args, entity='aipp', group=None)
        wandb.watch(model)
    else:
        logger = None

    # visualization callbacks
    #cb_2000 = VisualisationCallback(model, logger, 2000)
    #cb_2100 = VisualisationCallback(model, logger, 2100)
    cb_list = None
    checkpoint_path = args.checkpoint
    #write ground truth diagnostics
    #if pinn.rank == 0:
        #visualize_gt_diagnostics(cb_2000.dataset, 2000)
        #visualize_gt_diagnostics(cb_2100.dataset, 2100)


    pinn.fit(epochs=args.num_epochs,
             optimizer='Adam',
             learning_rate=args.learning_rate,
             pretraining=False,
             epochs_pt=30,
             lbfgs_finetuning=False,
             writing_cylcle=10,
             activate_annealing=args.annealing,
             logger=logger,
             checkpoint_path=checkpoint_path,
             restart=args.restart,
             callbacks=cb_list,
             pinn_path="best_model_" + args.name + '.pt'
             )
    #performance_callback.lock.value = 0
    #time.sleep(120)

