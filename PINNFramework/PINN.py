import torch
import torch.nn as nn
import numpy as np
from os.path import exists
from itertools import chain
from torch.utils.data import DataLoader
from .InitalCondition import InitialCondition
from .BoundaryCondition import BoundaryCondition, PeriodicBC, DirichletBC, NeumannBC, RobinBC
from .PDELoss import PDELoss
from .JoinedDataset import JoinedDataset
from .HPMLoss import HPMLoss
from torch.autograd import grad as grad
from PINNFramework.callbacks import CallbackList

try:
    import horovod.torch as hvd
except:
    print("Was not able to import Horovod. Thus Horovod support is not enabled")

# set initial seed for torch and numpy
torch.manual_seed(42)
np.random.seed(42)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class PINN(nn.Module):

    def __init__(self, model: torch.nn.Module, input_dimension: int, output_dimension: int,
                 pde_loss: PDELoss, initial_condition: InitialCondition, boundary_condition,
                 use_gpu=True, use_horovod=False,dataset_mode='min'):
        """
        Initializes an physics-informed neural network (PINN). A PINN consists of a model which represents the solution
        of the underlying partial differential equation(PDE) u, three loss terms representing initial (IC) and boundary
        condition(BC) and the PDE and a dataset which represents the bounded domain U.

        Args: 
            model : is the model which is trained to represent the underlying PDE
            input_dimension : represents the dimension of the input vector x
            output_dimension : represents the dimension of the solution u
            pde_loss: Instance of the PDELoss class. Represents the underlying PDE
            initial_condition: Instance of the InitialCondition class. Represents the initial condition
            boundary_condition (BoundaryCondition, list): Instance of the BoundaryCondition class or a list of instances
            of the BoundaryCondition class
            use_gpu: enables gpu usage
            use_horovod: enables horovod support

        """



        super(PINN, self).__init__()
        # checking if the model is a torch module more model checking should be possible
        self.use_gpu = use_gpu
        self.use_horovod = use_horovod
        self.rank = 0  # initialize rank 0 by default in order to make the fit method more flexible
        self.loss_log = {}
        if self.use_horovod:
            # Initialize Horovod
            hvd.init()
            # Pin GPU to be used to process local rank (one GPU per process)
            torch.cuda.set_device(hvd.local_rank())
            self.rank = hvd.rank()

        if isinstance(model, nn.Module):
            self.model = model
            if self.use_gpu:
                self.model.cuda()
                self.dtype = torch.cuda.FloatTensor
            else:
                self.dtype = torch.FloatTensor
        else:
            raise TypeError("Only models of type torch.nn.Module are allowed")

        # checking if the input dimension is well defined 
        if not type(input_dimension) is int:
            raise TypeError("Only integers are allowed as input dimension")
        elif input_dimension <= 0:
            raise ValueError("Input dimension has to be greater than zero")
        else:
            self.input_dimension = input_dimension

        # checking if the output dimension is well defined 
        if not type(output_dimension) is int:
            raise TypeError("Only integers are allowed as output dimension")
        elif input_dimension <= 0:
            raise ValueError("Input dimension has to be greater than zero")
        else:
            self.output_dimension = output_dimension

        if isinstance(pde_loss, PDELoss):
            self.pde_loss = pde_loss
            self.is_hpm = False
        else:
            raise TypeError("PDE loss has to be an instance of a PDE Loss class")

        if isinstance(pde_loss, HPMLoss):
            self.is_hpm = True

        if isinstance(initial_condition, InitialCondition):
            self.initial_condition = initial_condition
        else:
            raise TypeError("Initial condition has to be an instance of the InitialCondition class")

        joined_datasets = {
            initial_condition.name: initial_condition.dataset,
            pde_loss.name: pde_loss.dataset
        }
        self.loss_log[initial_condition.name] = float(0.0)  # adding initial condition to the loss_log
        self.loss_log[pde_loss.name] = float(0.0)

        if not self.is_hpm:
            if type(boundary_condition) is list:
                for bc in boundary_condition:
                    if not isinstance(bc, BoundaryCondition):
                        raise TypeError("Boundary Condition has to be an instance of the BoundaryCondition class ")
                    joined_datasets[bc.name] = bc.dataset
                    self.loss_log[bc.name] = float(0.0)
                self.boundary_condition = boundary_condition
            else:
                if isinstance(boundary_condition, BoundaryCondition):
                    self.boundary_condition = boundary_condition
                    joined_datasets[boundary_condition.name] = boundary_condition.dataset
                else:
                    raise TypeError("Boundary Condition has to be an instance of the BoundaryCondition class"
                                    "or a list of instances of the BoundaryCondition class")
        self.dataset = JoinedDataset(joined_datasets, dataset_mode)

    def loss_grad_std_wn(self, loss):
        device = torch.device("cuda" if self.use_gpu else "cpu")
        grad_ = torch.zeros((0), dtype=torch.float32, device=device)
        model_grads = grad(loss, self.model.parameters(), allow_unused=True, retain_graph=True)
        for elem in model_grads:
            if elem is not None:
                grad_ = torch.cat((grad_, elem.view(-1)))
        return torch.std(grad_)

    def forward(self, x):
        """
        Predicting the solution at given position x
        """
        return self.model(x)

    def save_model(self, pinn_path, hpm_path=None):
        """
        Saves the state dict of the models. Differs between HPM and Model

        Args:
            pinn_path: path where the pinn get stored
            hpm_path: path where the HPM get stored
        """
        if isinstance(self.pde_loss, HPMLoss):
            if hpm_path is None:
                raise ValueError("Saving path for the HPM has to be defined")
            torch.save(self.model.state_dict(), pinn_path)
            torch.save(self.pinn_loss.hpm_model.state_dict(), hpm_path)
        else:
            torch.save(self.model.state_dict(), pinn_path)

    def load_model(self, pinn_path, hpm_path=None):
        """
        Load the state dict of the models. Differs between HPM and Model

        Args:
            pinn_path: path from where the pinn get loaded
            hpm_path: path from where the HPM get loaded
        """
        if isinstance(self.pde_loss, HPMLoss):
            if hpm_path is None:
                raise ValueError("Loading path for the HPM has to be defined")
            self.model.load_state_dict(torch.load(pinn_path))
            self.pde_loss.hpm_model.load_state_dict(torch.load(hpm_path))
        else:
            self.model.load_state_dict(torch.load(pinn_path))

    def calculate_boundary_condition(self, boundary_condition: BoundaryCondition, training_data):
        """
        This function classifies the boundary condition and calculates the satisfaction

        Args:
            boundary_condition (BoundaryCondition) : boundary condition to be calculated
            training_data: training data used for evaluation
        """

        if isinstance(boundary_condition, PeriodicBC):
            # Periodic Boundary Condition
            if isinstance(training_data, list):
                if len(training_data) == 2:
                    return boundary_condition(training_data[0][0].type(self.dtype),
                                              training_data[1][0].type(self.dtype),
                                              self.model)
                else:
                    raise ValueError(
                        "The boundary condition {} has to be tuple of coordinates for lower and upper bound".
                            format(boundary_condition.name))
            else:
                raise ValueError("The boundary condition {} has to be tuple of coordinates for lower and upper bound".
                                 format(boundary_condition.name))
        if isinstance(boundary_condition, DirichletBC):
            # Dirchlet Boundary Condition
            if not isinstance(training_data, list):
                return boundary_condition(training_data.type(self.dtype)[0], self.model)
            else:
                raise ValueError("The boundary condition {} should be a tensor of coordinates not a tuple".
                                 format(boundary_condition.name))
        if isinstance(boundary_condition, NeumannBC):
            # Neumann Boundary Condition
            if not isinstance(training_data, list):
                return boundary_condition(training_data.type(self.dtype)[0], self.model)
            else:
                raise ValueError("The boundary condition {} should be a tensor of coordinates not a tuple".
                                 format(boundary_condition.name))
        if isinstance(boundary_condition, RobinBC):
            # Robin Boundary Condition
            if isinstance(training_data, list):
                if len(training_data) == 2:
                    return boundary_condition(training_data[0][0].type(self.dtype),
                                              training_data[1][0].type(self.dtype),
                                              self.model)
                else:
                    raise ValueError(
                        "The boundary condition {} has to be tuple of coordinates for lower and upper bound".
                            format(boundary_condition.name))
            else:
                raise ValueError("The boundary condition {} has to be tuple of coordinates for lower and upper bound".
                                 format(boundary_condition.name))

    def pinn_loss(self, training_data, annealing=False):
        """
        Function for calculating the PINN loss. The PINN Loss is a weighted sum of losses for initial and boundary
        condition and the residual of the PDE

        Args:
            training_data (Dictionary): Training Data for calculating the PINN loss in form of ta dictionary. The
            dictionary holds the training data for initial condition at the key "Initial_Condition" training data for
            the PDE at the key "PDE" and the data for the boundary condition under the name of the boundary condition
        """
        pinn_loss = 0
        # unpack training data
        # ============== PDE LOSS ============== "
        if type(training_data[self.pde_loss.name]) is not list:
            pde_loss = self.pde_loss(training_data[self.pde_loss.name][0].type(self.dtype), self.model)
            if annealing:
                std_pde = self.loss_grad_std_wn(pde_loss)
            pinn_loss = pinn_loss + pde_loss
            self.loss_log[self.pde_loss.name] += pde_loss
        else:
            raise ValueError("Training Data for PDE data is a single tensor consists of residual points ")

        # ============== INITIAL CONDITION ============== "
        if type(training_data[self.initial_condition.name]) is list:
            # initial condition loss
            if len(training_data[self.initial_condition.name]) == 2:
                ic_loss = self.initial_condition(
                    training_data[self.initial_condition.name][0][0].type(self.dtype),
                    self.model,
                    training_data[self.initial_condition.name][1][0].type(self.dtype)
                )
                self.loss_log[self.initial_condition.name] += ic_loss / self.initial_condition.weight
                if annealing:
                    std_ic = self.loss_grad_std_wn(ic_loss)
                    lambda_hat = std_pde / std_ic
                    self.initial_condition.weight = (1 - 0.5) * self.initial_condition.weight + 0.5 * lambda_hat
                pinn_loss = pinn_loss + ic_loss
            else:
                raise ValueError("Training Data for initial condition is a tuple (x,y) with x the  input coordinates"
                                 " and ground truth values y")
        else:
            raise ValueError("Training Data for initial condition is a tuple (x,y) with x the input coordinates"
                             " and ground truth values y")

        # ============== BOUNDARY CONDITION ============== "
        if not self.is_hpm:
            if isinstance(self.boundary_condition, list):
                for bc in self.boundary_condition:
                    bc_loss = self.calculate_boundary_condition(bc, training_data[bc.name])
                    self.loss_log[bc.name] += bc_loss / bc.weight
                    if annealing:
                        std_bc = self.loss_grad_std_wn(bc_loss)
                        lambda_hat = std_pde / std_bc
                        bc.weight = (1 - 0.5) * bc.weight + 0.5 * lambda_hat
                    pinn_loss = pinn_loss + bc_loss
            else:
                bc_loss = self.calculate_boundary_condition(self.boundary_condition,
                                                            training_data[self.boundary_condition.name])
                self.loss_log[self.boundary_condition.name] += bc_loss / self.boundary_condition.weight
                if annealing:
                    std_bc = self.loss_grad_std_wn(bc_loss)
                    lambda_hat = std_pde / std_bc
                    self.boundary_condition.weight = (1 - 0.5) * self.boundary_condition.weight + 0.5 * lambda_hat
                pinn_loss = pinn_loss + bc_loss

        # ============== Model specific losses  ============== "
        if hasattr(self.model, 'loss'):
            pinn_loss = pinn_loss + self.model.loss
            self.loss_log["model_loss_pinn"] += self.model.loss
        if self.is_hpm:
            if hasattr(self.pde_loss.model, 'loss'):
                pinn_loss = pinn_loss + self.model.loss
                self.loss_log["model_loss_hpm"] += self.model.loss

        return pinn_loss

    def write_checkpoint(self, checkpoint_path, epoch, pretraining, minimum_pinn_loss, optimizer):
        checkpoint =  {}
        checkpoint["epoch"] = epoch
        checkpoint["pretraining"] = pretraining
        checkpoint["minimum_pinn_loss"] = minimum_pinn_loss
        checkpoint["optimizer"] = optimizer.state_dict()
        checkpoint["weight_"+ self.initial_condition.name] = self.initial_condition.weight
        checkpoint["weight_" + self.pde_loss.name] = self.initial_condition.weight
        checkpoint["pinn_model"] = self.model.state_dict()
        if isinstance(self.boundary_condition, list):
            for bc in self.boundary_condition:
                checkpoint["weight_"+ bc.name] = bc.weight
        else:
            checkpoint["weight_" + self.boundary_condition.name] = self.boundary_condition.weight

        if self.is_hpm:
            checkpoint["hpm_model"] = self.pde_loss.hpm_model.state_dict()
        torch.save(checkpoint, checkpoint_path)



    def fit(self,
            epochs,
            checkpoint_path=None,
            restart=False,
            optimizer='Adam',
            learning_rate=1e-3,
            pretraining=False,
            epochs_pt=100,
            lbfgs_finetuning=True,
            writing_cylcle=30,
            writing_cycle_pt=10,
            save_model=True,
            pinn_path='best_model_pinn.pt',
            hpm_path='best_model_hpm.pt',
            logger=None,
            activate_annealing=False,
            annealing_cycle=100,
            callbacks=None):
        """
        Function for optimizing the parameters of the PINN-Model

        Args:
            epochs (int) : number of epochs used for training
            optimizer (String, torch.optim.Optimizer) : Optimizer used for training. At the moment only ADAM and LBFGS
            are supported by string command. It is also possible to give instances of torch optimizers as a parameter
            learning_rate: The learning rate of the optimizer
            pretraining: Activates seperate training on the initial condition at the beginning
            epochs_pt: defines the number of epochs for the pretraining
            lbfgs_finetuning: Enables LBFGS finetuning after main training
            writing_cylcle: defines the cylcus of model writing
            save_model: enables or disables checkpointing
            pinn_path: defines the path where the pinn get stored
            hpm_path: defines the path where the hpm get stored
            logger (Logger): tracks the convergence of all loss terms
            activate_annealing (Boolean): enables annealing
            annealing_cycle (int): defines the periodicity of using annealing
            callbacks (CallbackList): is a list of callbacks which are called at the end of a writing cycle. Can be used
            for different purposes e.g. early stopping, visualization, model state logging etc.
            checkpoint_path (string) : path to the checkpoint
            restart (integer) : defines if checkpoint will be used (False) or will be overwritten (True)


        """
        # checking if callbacks are a instance of CallbackList
        if callbacks is not None:
            if not isinstance(callbacks, CallbackList):
                raise ValueError("Callbacks has to be a instance of CallbackList but type {} was found".
                                 format(type(callbacks)))

        if isinstance(self.pde_loss, HPMLoss):
            params = list(self.model.parameters()) + list(self.pde_loss.hpm_model.parameters())
            named_parameters = chain(self.model.named_parameters(), self.pde_loss.hpm_model.named_parameters())
            if self.use_horovod and lbfgs_finetuning:
                raise ValueError("LBFGS Finetuning is not possible with horovod")
            if optimizer == 'Adam':
                optim = torch.optim.Adam(params, lr=learning_rate)
            elif optimizer == 'LBFGS':
                if self.use_horovod:
                    raise TypeError("LBFGS is not supported with Horovod")
                else:
                    optim = torch.optim.LBFGS(params, lr=learning_rate)
            else:
                optim = optimizer

            if lbfgs_finetuning and not self.use_horovod:
                lbfgs_optim = torch.optim.LBFGS(params, lr=0.9)

                def closure():
                    lbfgs_optim.zero_grad()
                    pinn_loss = self.pinn_loss(training_data)
                    pinn_loss.backward()
                    return pinn_loss
        else:
            named_parameters = self.model.named_parameters()
            if optimizer == 'Adam':
                optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            elif optimizer == 'LBFGS':
                optim = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate)
            else:
                optim = optimizer

            if lbfgs_finetuning and not self.use_horovod:
                lbfgs_optim = torch.optim.LBFGS(self.model.parameters(), lr=0.9)

                def closure():
                    lbfgs_optim.zero_grad()
                    pinn_loss = self.pinn_loss(training_data)
                    pinn_loss.backward()
                    return pinn_loss

        minimum_pinn_loss = float("inf")
        if self.use_horovod:
            # Partition dataset among workers using DistributedSampler
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset, num_replicas=hvd.size(), rank=hvd.rank()
            )
            data_loader = DataLoader(self.dataset, batch_size=1, sampler=train_sampler, worker_init_fn=worker_init_fn)
            optim = hvd.DistributedOptimizer(optim, named_parameters=named_parameters)
            if pretraining:
                train_sampler_pt = torch.utils.data.distributed.DistributedSampler(
                    self.initial_condition.dataset, num_replicas=hvd.size(), rank=hvd.rank()
                )
                data_loader_pt = DataLoader(self.initial_condition.dataset,
                                            batch_size=None,
                                            sampler=train_sampler_pt,
                                            worker_init_fn=worker_init_fn)
            # Broadcast parameters from rank 0 to all other processes.
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            if isinstance(self.pde_loss, HPMLoss):
                hvd.broadcast_parameters(self.pinn_loss.hpm_model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optim, root_rank=0)

        else:
            data_loader = DataLoader(self.dataset, batch_size=1, worker_init_fn=worker_init_fn)
            data_loader_pt = DataLoader(self.initial_condition.dataset, batch_size=None, worker_init_fn=worker_init_fn)

        start_epoch = 0

        # load checkpoint routine if a checkpoint path is set and its allowed to not overwrite the checkpoint
        if checkpoint_path is not None:
            if not exists(checkpoint_path) and not restart:
                raise FileNotFoundError(
                    "Checkpoint path {} do not exists. Please change the path to a existing checkpoint"
                    "or change the restart flag to true in order to create a new checkpoint"
                    .format(checkpoint_path))
        if checkpoint_path is not None and not restart:
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint["epoch"]
            pretraining = checkpoint["pretraining"]
            self.initial_condition.weight = checkpoint["weight_" + self.initial_condition.name]
            self.pde_loss.weight = checkpoint["weight_" + self.pde_loss.name]
            if isinstance(self.boundary_condition, list):
                for bc in self.boundary_condition:
                    bc.weight = checkpoint["weight_" + bc.name]
            else:
                self.boundary_condition.weight = checkpoint["weight_" + self.boundary_condition.name]

            self.model.load_state_dict(checkpoint["pinn_model"])
            if self.is_hpm:
                self.pde_loss.hpm_model.load_state_dict(checkpoint["hpm_model"])

            optim.load_state_dict(checkpoint['optimizer'])
            minimum_pinn_loss = checkpoint["minimum_pinn_loss"]
            print("Checkpoint Loaded")

        print("===== Pretraining =====")
        if pretraining:
            for epoch in range(start_epoch, epochs_pt):
                for x, y in data_loader_pt:
                    optim.zero_grad()
                    ic_loss = self.initial_condition(model=self.model, x=x.type(self.dtype), gt_y=y.type(self.dtype))
                    ic_loss.backward()
                    optim.step()
                    if not self.rank:
                        if not (epoch + 1) % writing_cycle_pt and checkpoint_path is not None:
                            self.write_checkpoint(checkpoint_path, epoch, True, minimum_pinn_loss, optim)
                if not self.rank:
                    print("IC Loss {} Epoch {} from {}".format(ic_loss, epoch+1, epochs_pt))
        print("===== Main training =====")
        for epoch in range(start_epoch, epochs):
            np.random.seed(42 + epoch)
            batch_counter = 0.
            pinn_loss_sum = 0.
            for idx, training_data in enumerate(data_loader):
                do_annealing = activate_annealing and (idx == 0) and not (epoch + 1) % annealing_cycle
                optim.zero_grad()
                pinn_loss = self.pinn_loss(training_data, do_annealing)
                pinn_loss.backward()
                optim.step()
                pinn_loss_sum += pinn_loss
                batch_counter += 1
            if not self.rank:
                print("PINN Loss {} Epoch {} from {}".format(pinn_loss_sum / batch_counter, epoch, epochs), flush=True)
                if logger is not None and not epoch % writing_cylcle:
                    logger.log_scalar(scalar=pinn_loss_sum / batch_counter, name=" Weighted PINN Loss", epoch=epoch)
                    logger.log_scalar(scalar=sum(self.loss_log.values()), name=" Non-Weighted PINN Loss", epoch=epoch)
                    # Log values of the loss terms
                    for key, value in self.loss_log.items():
                        logger.log_scalar(scalar=value / batch_counter, name=key, epoch=epoch)

                    # Log weights of loss terms separately
                    logger.log_scalar(scalar=self.initial_condition.weight,
                                      name=self.initial_condition.name + "_weight",
                                      epoch=epoch)
                    if not self.is_hpm:
                        if isinstance(self.boundary_condition, list):
                            for bc in self.boundary_condition:
                                logger.log_scalar(scalar=bc.weight,
                                                  name=bc.name + "_weight",
                                                  epoch=epoch)
                        else:
                            logger.log_scalar(scalar=self.boundary_condition.weight,
                                              name=self.boundary_condition.name + "_weight",
                                              epoch=epoch)
                    callbacks(epoch=epoch)
                # saving routine
                if (pinn_loss_sum / batch_counter < minimum_pinn_loss) and save_model:
                    self.save_model(pinn_path, hpm_path)
                    minimum_pinn_loss = pinn_loss_sum / batch_counter

                # reset loss log after the end of the epoch
                for key in self.loss_log.keys():
                    self.loss_log[key] = float(0)

                # writing checkpoint
                if not (epoch + 1) % writing_cylcle and checkpoint_path is not None:
                    self.write_checkpoint(checkpoint_path, epoch, False, minimum_pinn_loss, optim)

        if lbfgs_finetuning:
            lbfgs_optim.step(closure)
            print("After LBFGS-B: PINN Loss {} Epoch {} from {}".format(pinn_loss, epoch+1, epochs))
            if (pinn_loss < minimum_pinn_loss) and not (epoch % writing_cylcle) and save_model:
                self.save_model(pinn_path, hpm_path)
