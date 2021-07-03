import torch
import torch.nn as nn
from itertools import chain
from torch.utils.data import DataLoader
from .InitalCondition import InitialCondition
from .BoundaryCondition import BoundaryCondition, PeriodicBC, DirichletBC, NeumannBC, RobinBC
from .PDELoss import PDELoss
from .JoinedDataset import JoinedDataset
from .HPMLoss import HPMLoss

try:
    import horovod.torch as hvd
except:
    print("Was not able to import Horovod. Thus Horovod support is not enabled")

class PINN(nn.Module):

    def __init__(self, model: torch.nn.Module, input_dimension: int, output_dimension: int,
                 pde_loss: PDELoss, initial_condition: InitialCondition, boundary_condition,
                 use_gpu=True, use_horovod=False):
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
        self.rank = 0 # initialize rank 0 by default in order to make the fit method more flexible
        if self.use_horovod:
            # Initialize Horovod
            hvd.init()
            # Pin GPU to be used to process local rank (one GPU per process)
            torch.cuda.set_device(hvd.local_rank())
            self.rank = hvd.rank()

        if not isinstance(model, nn.Module):
            raise TypeError("Only models of type torch.nn.Module are allowed")

        self.model = model
        if self.use_gpu:
            self.model.cuda()
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
        # checking if the input dimension is well defined
        if type(input_dimension) is not int:
            raise TypeError("Only integers are allowed as input dimension")
        elif input_dimension <= 0:
            raise ValueError("Input dimension has to be greater than zero")
        else:
            self.input_dimension = input_dimension

        # checking if the output dimension is well defined
        if type(output_dimension) is not int:
            raise TypeError("Only integers are allowed as output dimension")
        elif input_dimension <= 0:
            raise ValueError("Input dimension has to be greater than zero")
        else:
            self.output_dimension = output_dimension

        if not isinstance(pde_loss, PDELoss):
            raise TypeError("PDE loss has to be an instance of a PDE Loss class")

        self.pde_loss = pde_loss
        self.is_hpm = False
        if isinstance(pde_loss,HPMLoss):
            self.is_hpm = True

        if isinstance(initial_condition, InitialCondition):
            self.initial_condition = initial_condition
        else:
            raise TypeError("Initial condition has to be an instance of the InitialCondition class")

        joined_datasets = {"Initial_Condition": initial_condition.dataset, "PDE": pde_loss.dataset}
        if not self.is_hpm:
            if type(boundary_condition) is list:
                for bc in boundary_condition:
                    if not isinstance(bc, BoundaryCondition):
                        raise TypeError("Boundary Condition has to be an instance of the BoundaryCondition class ")
                    self.boundary_condition = boundary_condition
                    joined_datasets[bc.name] = bc.dataset

            elif isinstance(boundary_condition, BoundaryCondition):
                self.boundary_condition = boundary_condition
            else:
                raise TypeError("Boundary Condition has to be an instance of the BoundaryCondition class"
                                "or a list of instances of the BoundaryCondition class")
        self.dataset = JoinedDataset(joined_datasets)

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
            torch.save(self.pde_loss.hpm_model.state_dict(), hpm_path)
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
            if isinstance(training_data, list) and len(training_data) == 2:
                return boundary_condition(training_data[0][0].type(self.dtype),
                                          training_data[1][0].type(self.dtype),
                                          self.model)
            else:
                raise ValueError(
                    "The boundary condition {} has to be tuple of coordinates for lower and upper bound".
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
            if isinstance(training_data, list) and len(training_data) == 2:
                return boundary_condition(training_data[0][0].type(self.dtype),
                                          training_data[1][0].type(self.dtype),
                                          self.model)
            else:
                raise ValueError(
                    "The boundary condition {} has to be tuple of coordinates for lower and upper bound".
                    format(boundary_condition.name))

    def pinn_loss(self, training_data):
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
        if type(training_data["Initial_Condition"]) is not list:
            raise ValueError("Training Data for initial condition is a tuple (x,y) with x the  input coordinates"
                             " and ground truth values y")

            # initial condition loss
        if len(training_data["Initial_Condition"]) == 2:
            pinn_loss += self.initial_condition(
                training_data["Initial_Condition"][0][0].type(self.dtype),
                self.model,
                training_data["Initial_Condition"][1][0].type(self.dtype),
            )

        else:
            raise ValueError("Training Data for initial condition is a tuple (x,y) with x the  input coordinates"
                             " and ground truth values y")
        if type(training_data["PDE"]) is not list:
            pinn_loss += self.pde_loss(
                training_data["PDE"][0].type(self.dtype), self.model
            )

        else:
            raise ValueError("Training Data for PDE data is a single tensor consists of residual points ")
        if not self.is_hpm:
            if isinstance(self.boundary_condition, list):
                for bc in self.boundary_condition:
                    pinn_loss += self.calculate_boundary_condition(bc, training_data[bc.name])
            else:
                pinn_loss += self.calculate_boundary_condition(
                    self.boundary_condition,
                    training_data[self.boundary_condition.name],
                )

        return pinn_loss

    def fit(self, epochs, optimizer='Adam', learning_rate=1e-3, lbfgs_finetuning=True,
            writing_cylcle= 30, save_model=True, pinn_path='best_model_pinn.pt', hpm_path='best_model_hpm.pt'):
        """
        Function for optimizing the parameters of the PINN-Model

        Args:
            epochs (int) : number of epochs used for training
            optimizer (String, torch.optim.Optimizer) : Optimizer used for training. At the moment only ADAM and LBFGS
            are supported by string command. It is also possible to give instances of torch optimizers as a parameter
            learning_rate: The learning rate of the optimizer
            lbfgs_finetuning: Enables LBFGS finetuning after main training
            writing_cylcle: defines the cylcus of model writing
            save_model: enables or disables checkpointing
            pinn_path: defines the path where the pinn get stored
            hpm_path: defines the path where the hpm get stored

        """
        if isinstance(self.pde_loss, HPMLoss):
            params = list(self.model.parameters()) + list(self.pde_loss.hpm_model.parameters())
            named_parameters = chain(self.model.named_parameters(),self.pde_loss.hpm_model.named_parameters())
            if self.use_horovod  and lbfgs_finetuning:
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
            self.dataset, num_replicas=hvd.size(), rank=hvd.rank())
            data_loader = DataLoader(self.dataset, batch_size=1,sampler=train_sampler)
            optim = hvd.DistributedOptimizer(optim, named_parameters=named_parameters)
            # Broadcast parameters from rank 0 to all other processes.
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            if isinstance(self.pde_loss, HPMLoss):
                hvd.broadcast_parameters(self.pinn_loss.hpm_model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optim, root_rank=0)

        else:
            data_loader = DataLoader(self.dataset, batch_size=1)

        for epoch in range(epochs):
            for training_data in data_loader:
                training_data = training_data
                optim.zero_grad()
                pinn_loss = self.pinn_loss(training_data)
                pinn_loss.backward()
                if not self.rank:
                    print("PINN Loss {} Epoch {} from {}".format(pinn_loss, epoch, epochs))
                optim.step()
            if (pinn_loss < minimum_pinn_loss) and not (epoch % writing_cylcle) and save_model and not self.rank:
                self.save_model(pinn_path, hpm_path)
                minimum_pinn_loss = pinn_loss

        if lbfgs_finetuning:
            lbfgs_optim.step(closure)
            print("After LBFGS-B: PINN Loss {} Epoch {} from {}".format(pinn_loss, epoch, epochs))
            if (pinn_loss < minimum_pinn_loss) and not (epoch % writing_cylcle) and save_model:
                self.save_model(pinn_path, hpm_path)
