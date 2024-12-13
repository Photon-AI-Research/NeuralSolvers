from typing import Union, List

import torch
import torch.nn as nn
import numpy as np
from os.path import exists
from datetime import datetime
from itertools import chain
from torch.utils.data import DataLoader
from NeuralSolvers.pinn.datasets.InitalCondition import InitialCondition
from NeuralSolvers.pinn.datasets.BoundaryCondition import BoundaryCondition, PeriodicBC, DirichletBC, NeumannBC, RobinBC, TimeDerivativeBC
from .PDELoss import PDELoss
from NeuralSolvers.JoinedDataset import JoinedDataset
from .HPMLoss import HPMLoss
from NeuralSolvers.samplers.Adaptive_Sampler import AdaptiveSampler
from torch.autograd import grad as grad
from NeuralSolvers.callbacks.Callback import CallbackList
#from pyevtk.hl import imageToVTK

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
                 device='cpu', use_horovod=False, dataset_mode='min'):
        """
        Initializes a physics-informed neural network (PINN).

        Args:
            model (torch.nn.Module): The model representing the PDE solution.
            input_dimension (int): Dimension of the input vector x.
            output_dimension (int): Dimension of the solution u.
            pde_loss (PDELoss): Instance representing the underlying PDE.
            initial_condition (InitialCondition): Instance representing the initial condition.
            boundary_condition (BoundaryCondition or list): Boundary condition(s).
            device (str): ML accelerator ['cpu', 'gpu', 'mps', etc.].
            use_horovod (bool): Enable Horovod support.
            dataset_mode (str): Behavior of the joined dataset.
        """
        super(PINN, self).__init__()
        self.device = torch.device(device)
        self.use_horovod = use_horovod
        self.rank = self._initialize_horovod() if use_horovod else 0
        self.loss_gradients_storage = {}
        self.loss_log = {} if self.rank == 0 else None

        self._validate_and_set_model(model)
        self._validate_and_set_dimensions(input_dimension, output_dimension)
        self._validate_and_set_pde_loss(pde_loss)
        self._validate_and_set_initial_condition(initial_condition)
        self._validate_and_set_boundary_conditions(boundary_condition)

        joined_datasets = self._create_joined_datasets(initial_condition, pde_loss, boundary_condition)
        self.dataset = JoinedDataset(joined_datasets, dataset_mode)

    def _initialize_horovod(self):
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
        return hvd.rank()

    def _validate_and_set_model(self, model):
        if not isinstance(model, nn.Module):
            raise TypeError("Model must be an instance of torch.nn.Module")
        self.model = model
        self.model.to(self.device)
        self.dtype = torch.float32

    def _validate_and_set_dimensions(self, input_dimension, output_dimension):
        for dim, name in [(input_dimension, "input"), (output_dimension, "output")]:
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"{name.capitalize()} dimension must be a positive integer")
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

    def _validate_and_set_pde_loss(self, pde_loss):
        if not isinstance(pde_loss, PDELoss):
            raise TypeError("PDE loss must be an instance of PDELoss")
        self.pde_loss = pde_loss
        self.is_hpm = isinstance(pde_loss, HPMLoss)
        if self.is_hpm:
            self.pde_loss.hpm_model.to(self.device)
        if isinstance(pde_loss.geometry.sampler, AdaptiveSampler):
            self.pde_loss.geometry.sampler.device = self.device
        if not len(pde_loss.geometry):
            raise ValueError("Geometry is empty")

    def _validate_and_set_initial_condition(self, initial_condition):
        if not isinstance(initial_condition, InitialCondition):
            raise TypeError("Initial condition must be an instance of InitialCondition")
        if not len(initial_condition.dataset):
            raise ValueError("Initial condition dataset is empty")
        self.initial_condition = initial_condition

    def _validate_and_set_boundary_conditions(self, boundary_condition):
        if boundary_condition is None:
            boundary_condition = None
            return

        if isinstance(boundary_condition, list):
            for bc in boundary_condition:
                self._validate_single_boundary_condition(bc)
            self.boundary_condition = boundary_condition
        elif isinstance(boundary_condition, BoundaryCondition):
            self._validate_single_boundary_condition(boundary_condition)
            self.boundary_condition = [boundary_condition]
        else:
            raise TypeError("Boundary condition must be a BoundaryCondition instance or a list of instances")

    def _validate_single_boundary_condition(self, bc):
        if not isinstance(bc, BoundaryCondition):
            raise TypeError("Each boundary condition must be an instance of BoundaryCondition")
        if not len(bc.dataset):
            raise ValueError(f"Boundary condition dataset for {bc.name} is empty")

    def _create_joined_datasets(self, initial_condition, pde_loss, boundary_condition):
        joined_datasets = {
            initial_condition.name: initial_condition.dataset,
            pde_loss.name: pde_loss.geometry
        }
        if self.rank == 0:
            self.loss_log[initial_condition.name] = 0.0
            self.loss_log[pde_loss.name] = 0.0
            if hasattr(self.model, 'loss'):
                self.loss_log["model_loss_pinn"] = 0.0

        if not self.is_hpm:
            for bc in self.boundary_condition:
                joined_datasets[bc.name] = bc.dataset
                if self.rank == 0:
                    self.loss_log[bc.name] = 0.0

        return joined_datasets


    def loss_gradients(self, loss):
        device = self.device
        grad_ = torch.zeros((0), dtype=torch.float32, device=device)
        model_grads = grad(loss, self.model.parameters(), allow_unused=True, retain_graph=True)
        for elem in model_grads:
            if elem is not None:
                grad_ = torch.cat((grad_, elem.view(-1)))
        return grad_

    def forward(self, x):
        """
        Predicting the solution at given pos
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

    def calculate_boundary_condition(self, boundary_condition: BoundaryCondition,
                                     training_data: Union[torch.Tensor, List[torch.Tensor]]):
        """
        Classify the boundary condition and calculate its satisfaction.

        Args:
            boundary_condition (BoundaryCondition): Boundary condition to be calculated.
            training_data (Union[torch.Tensor, List[torch.Tensor]]): Training data used for evaluation.

        Returns:
            torch.Tensor: The calculated boundary condition loss.

        Raises:
            ValueError: If the training data format is incorrect for the given boundary condition type.
        """
        bc_type = type(boundary_condition)
        bc_handlers = {
            PeriodicBC: self._handle_periodic_bc,
            DirichletBC: self._handle_dirichlet_bc,
            NeumannBC: self._handle_neumann_bc,
            RobinBC: self._handle_robin_bc,
            TimeDerivativeBC: self._handle_time_derivative_bc
        }

        handler = bc_handlers.get(bc_type)
        if handler is None:
            raise ValueError(f"Unsupported boundary condition type: {bc_type}")

        return handler(boundary_condition, training_data)

    def _handle_periodic_bc(self, bc: PeriodicBC, training_data: List[torch.Tensor]):
        self._validate_training_data(training_data, expected_length=2, bc_name=bc.name)
        return bc(training_data[0][0].type(self.dtype), training_data[1][0].type(self.dtype), self.model)

    def _handle_dirichlet_bc(self, bc: DirichletBC, training_data: torch.Tensor):
        self._validate_training_data(training_data, expected_type=torch.Tensor, bc_name=bc.name)
        return bc(training_data.type(self.dtype)[0], self.model)

    def _handle_neumann_bc(self, bc: NeumannBC, training_data: torch.Tensor):
        self._validate_training_data(training_data, expected_type=torch.Tensor, bc_name=bc.name)
        return bc(training_data.type(self.dtype)[0], self.model)

    def _handle_robin_bc(self, bc: RobinBC, training_data: List[torch.Tensor]):
        self._validate_training_data(training_data, expected_length=2, bc_name=bc.name)
        return bc(training_data[0][0].type(self.dtype), training_data[1][0].type(self.dtype), self.model)

    def _handle_time_derivative_bc(self, bc: TimeDerivativeBC, training_data: List[torch.Tensor]):
        self._validate_training_data(training_data, expected_length=2, bc_name=bc.name)
        return bc(training_data[0][0].type(self.dtype), training_data[1][0].type(self.dtype), self.model)

    def _validate_training_data(self, training_data: Union[torch.Tensor, List[torch.Tensor]],
                                expected_type: type = list, expected_length: int = None, bc_name: str = ""):
        if not isinstance(training_data, expected_type):
            raise ValueError(f"The boundary condition {bc_name} expects {expected_type.__name__} training data, "
                             f"but got {type(training_data).__name__}")

        if expected_length and len(training_data) != expected_length:
            raise ValueError(f"The boundary condition {bc_name} expects a tuple of {expected_length} elements, "
                             f"but got {len(training_data)}")

    def inverse_dirichlet_annealing(self, alpha=0.5):
        # calculating maximum std
        stds = [torch.std(grad_) for grad_ in self.loss_gradients_storage.values()]
        maximum_std = max(stds)

        # annealing initial condition
        lambda_ic_head = maximum_std / torch.std(self.loss_gradients_storage[self.initial_condition.name])
        self.initial_condition.weight = 0.5 * self.initial_condition.weight + 0.5 * lambda_ic_head

        # annealing boundary condition
        if isinstance(self.boundary_condition, list):
            for bc in self.boundary_condition:
                # annealing initial condition
                lambda_bc_head = maximum_std / torch.std(self.loss_gradients_storage[bc.name])
                bc.weight = 0.5 * bc.weight + 0.5 * lambda_bc_head
        else:
            lambda_bc_head = maximum_std / torch.std(self.loss_gradients_storage[self.boundary_condition.name])
            self.boundary_condition.weight = 0.5 * self.boundary_condition.weight + 0.5 * lambda_bc_head

        # annealing pde loss
        lambda_pde_head = maximum_std / torch.std(self.loss_gradients_storage[self.pde_loss.name])
        self.pde_loss.weight = 0.5 * self.pde_loss.weight + 0.5 * lambda_pde_head

    def standard_learning_rate_annealing(self, alpha=0.9):
        # calculating maximum std
        maximum_residual = torch.max(torch.abs(self.loss_gradients_storage[self.pde_loss.name]))

        # annealing initial condition
        lambda_ic_head = maximum_residual / torch.mean(
            torch.abs(self.loss_gradients_storage[self.initial_condition.name]))
        self.initial_condition.weight = (1 - alpha) * self.initial_condition.weight + (alpha * lambda_ic_head)

        # annealing boundary condition
        if isinstance(self.boundary_condition, list):
            for bc in self.boundary_condition:
                # annealing initial condition
                lambda_bc_head = maximum_residual / torch.mean(torch.abs(self.loss_gradients_storage[bc.name]))
                bc.weight = (1 - alpha) * bc.weight + alpha * lambda_bc_head
        else:
            lambda_bc_head = maximum_residual / torch.mean(
                torch.abs(self.loss_gradients_storage[self.boundary_condition.name]))
            self.boundary_condition.weight = (1 - alpha) * self.boundary_condition.weight + alpha * lambda_bc_head

    def pinn_loss(self, training_data, track_gradient=False, annealing=False):
        """
        Calculate the PINN loss as a weighted sum of losses for initial condition, boundary condition, and PDE residual.

        Args:
            training_data (dict): Dictionary containing training data for different components.
            track_gradient (bool): If True, track gradients of loss terms.
            annealing (bool): If True, activate automatic balancing of loss terms.

        Returns:
            torch.Tensor: The total PINN loss.
        """
        self._initialize_gradient_storage(track_gradient, annealing)
        pinn_loss = 0.0

        # PDE Loss
        pinn_loss += self._calculate_pde_loss(training_data, track_gradient, annealing)

        # Initial Condition Loss
        pinn_loss += self._calculate_initial_condition_loss(training_data, track_gradient, annealing)

        # Boundary Condition Loss
        if not self.is_hpm:
            pinn_loss += self._calculate_boundary_condition_loss(training_data, track_gradient, annealing)

        # Model-specific losses
        pinn_loss += self._calculate_model_specific_losses()

        if annealing:
            self.inverse_dirichlet_annealing(alpha=0.5)

        return pinn_loss

    def _initialize_gradient_storage(self, track_gradient, annealing):
        if track_gradient or annealing:
            self.loss_gradients_storage = {}

    def _calculate_pde_loss(self, training_data, track_gradient, annealing):
        pde_data = training_data[self.pde_loss.name]
        if not isinstance(pde_data, list):
            pde_loss = self.pde_loss(pde_data[0].type(self.dtype), self.model)
            self._update_gradient_storage(self.pde_loss.name, pde_loss, track_gradient, annealing)
            self._update_loss_log(self.pde_loss.name, pde_loss)
            return self.pde_loss.weight * pde_loss
        else:
            raise ValueError("PDE training data should be a single tensor of residual points.")

    def _calculate_initial_condition_loss(self, training_data, track_gradient, annealing):
        ic_data = training_data[self.initial_condition.name]
        if isinstance(ic_data, list) and len(ic_data) == 2:
            ic_loss = self.initial_condition(
                ic_data[0][0].type(self.dtype),
                self.model,
                ic_data[1][0].type(self.dtype)
            )
            self._update_gradient_storage(self.initial_condition.name, ic_loss, track_gradient, annealing)
            self._update_loss_log(self.initial_condition.name, ic_loss)
            return ic_loss * self.initial_condition.weight
        else:
            raise ValueError(
                "Initial condition data should be a tuple (x, y) of input coordinates and ground truth values.")

    def _calculate_boundary_condition_loss(self, training_data, track_gradient, annealing):
        bc_loss = 0.0
        boundary_conditions = self.boundary_condition if isinstance(self.boundary_condition, list) else [
            self.boundary_condition]

        for bc in boundary_conditions:
            bc_data = training_data[bc.name]
            current_bc_loss = self.calculate_boundary_condition(bc, bc_data)
            self._update_gradient_storage(bc.name, current_bc_loss, track_gradient, annealing)
            self._update_loss_log(bc.name, current_bc_loss)
            bc_loss += current_bc_loss * bc.weight

        return bc_loss

    def _calculate_model_specific_losses(self):
        additional_loss = 0.0
        if hasattr(self.model, 'loss'):
            additional_loss += self.model.loss
            self._update_loss_log("model_loss_pinn", self.model.loss)

        if self.is_hpm and hasattr(self.pde_loss.hpm_model, 'loss'):
            additional_loss += self.pde_loss.hpm_model.loss
            self._update_loss_log("model_loss_hpm", self.pde_loss.hpm_model.loss)

        return additional_loss

    def _update_gradient_storage(self, name, loss, track_gradient, annealing):
        if track_gradient or annealing:
            self.loss_gradients_storage[name] = self.loss_gradients(loss)

    def _update_loss_log(self, name, loss):
        if self.rank == 0:
            self.loss_log[name] = self.loss_log.get(name, 0.0) + loss

    def write_checkpoint(self, checkpoint_path, epoch, pretraining, minimum_pinn_loss, optimizer):
        checkpoint = {}
        checkpoint["epoch"] = epoch
        checkpoint["pretraining"] = pretraining
        checkpoint["minimum_pinn_loss"] = minimum_pinn_loss
        checkpoint["optimizer"] = optimizer.state_dict()
        checkpoint["weight_" + self.initial_condition.name] = self.initial_condition.weight
        checkpoint["weight_" + self.pde_loss.name] = self.initial_condition.weight
        checkpoint["pinn_model"] = self.model.state_dict()
        if isinstance(self.boundary_condition, list):
            for bc in self.boundary_condition:
                checkpoint["weight_" + bc.name] = bc.weight
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
            writing_cycle=30,
            writing_cycle_pt=30,
            save_model=True,
            pinn_path='best_model_pinn.pt',
            hpm_path='best_model_hpm.pt',
            logger=None,
            track_gradient=False,
            activate_annealing=False,
            annealing_cycle=100,
            callbacks=None):
        """
        Function for optimizing the parameters of the PINN-Model
        """
        self._validate_fit_inputs(callbacks)
        optimizer = self._setup_optimizer(optimizer, learning_rate, lbfgs_finetuning)
        data_loader, data_loader_pt = self._setup_data_loaders()
        start_epoch, minimum_pinn_loss = self._load_checkpoint(checkpoint_path, restart, optimizer)

        if pretraining:
            self._perform_pretraining(optimizer, epochs_pt, data_loader_pt, writing_cycle_pt, checkpoint_path)

        self._perform_main_training(
            optimizer, epochs, start_epoch, data_loader, writing_cycle, checkpoint_path,
            save_model, pinn_path, hpm_path, logger, track_gradient, activate_annealing,
            annealing_cycle, callbacks, lbfgs_finetuning, minimum_pinn_loss
        )

    def _validate_fit_inputs(self, callbacks):
        if callbacks is not None and not isinstance(callbacks, CallbackList):
            raise ValueError("Callbacks must be an instance of CallbackList")

    def _setup_optimizer(self, optimizer, learning_rate, lbfgs_finetuning):
        if isinstance(self.pde_loss, HPMLoss):
            params = list(self.model.parameters()) + list(self.pde_loss.hpm_model.parameters())
            named_parameters = chain(self.model.named_parameters(), self.pde_loss.hpm_model.named_parameters())
        else:
            params = self.model.parameters()
            named_parameters = self.model.named_parameters()

        if optimizer == 'Adam':
            optim = torch.optim.Adam(params, lr=learning_rate)
        elif optimizer == 'LBFGS':
            optim = torch.optim.LBFGS(params, lr=learning_rate)
        else:
            optim = optimizer

        if self.use_horovod:
            optim = hvd.DistributedOptimizer(optim, named_parameters=named_parameters)

        return optim

    def _setup_data_loaders(self):
        if self.use_horovod:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset, num_replicas=hvd.size(), rank=hvd.rank()
            )
            data_loader = DataLoader(self.dataset, batch_size=1, sampler=train_sampler, worker_init_fn=worker_init_fn)
            train_sampler_pt = torch.utils.data.distributed.DistributedSampler(
                self.initial_condition.dataset, num_replicas=hvd.size(), rank=hvd.rank()
            )
            data_loader_pt = DataLoader(self.initial_condition.dataset,
                                        batch_size=None,
                                        sampler=train_sampler_pt,
                                        worker_init_fn=worker_init_fn)
        else:
            data_loader = DataLoader(self.dataset, batch_size=1, worker_init_fn=worker_init_fn)
            data_loader_pt = DataLoader(self.initial_condition.dataset, batch_size=None, worker_init_fn=worker_init_fn)

        return data_loader, data_loader_pt

    def _load_checkpoint(self, checkpoint_path, restart, optimizer):
        start_epoch = 0
        minimum_pinn_loss = float("inf")

        if checkpoint_path is not None and not restart:
            if not exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist.")

            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint["epoch"]
            self._load_weights_from_checkpoint(checkpoint)
            optimizer.load_state_dict(checkpoint['optimizer'])
            minimum_pinn_loss = checkpoint["minimum_pinn_loss"]
            print("Checkpoint Loaded", flush=True)
        else:
            print("Checkpoint not loaded", flush=True)

        return start_epoch, minimum_pinn_loss

    def _load_weights_from_checkpoint(self, checkpoint):
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

    def _perform_pretraining(self, optimizer, epochs_pt, data_loader_pt, writing_cycle_pt, checkpoint_path):
        print("===== Pretraining =====")
        for epoch in range(epochs_pt):
            epoch_start_time = datetime.now()
            for x, y in data_loader_pt:
                optimizer.zero_grad()
                ic_loss = self.initial_condition(model=self.model, x=x.type(self.dtype), gt_y=y.type(self.dtype))
                ic_loss.backward()
                optimizer.step()

            self._log_pretraining_progress(epoch, epochs_pt, ic_loss, epoch_start_time, writing_cycle_pt,
                                           checkpoint_path, optimizer)

    def _log_pretraining_progress(self, epoch, epochs_pt, ic_loss, epoch_start_time, writing_cycle_pt, checkpoint_path,
                                  optimizer):
        if not self.rank and not (epoch + 1) % writing_cycle_pt and checkpoint_path is not None:
            self.write_checkpoint(checkpoint_path, epoch, True, float("inf"), optimizer)
        if not self.rank:
            epoch_end_time = datetime.now()
            time_taken = (epoch_end_time - epoch_start_time).total_seconds()
            print(
                f"[{epoch_end_time}]:Epoch {epoch + 1:2d}/{epochs_pt} | IC Loss {ic_loss:.15f} | Epoch Duration {time_taken:.5f}")

    def _perform_main_training(self, optimizer, epochs, start_epoch, data_loader, writing_cycle, checkpoint_path,
                               save_model, pinn_path, hpm_path, logger, track_gradient, activate_annealing,
                               annealing_cycle, callbacks, lbfgs_finetuning, minimum_pinn_loss):
        print("===== Main training =====")
        for epoch in range(start_epoch, epochs):
            epoch_start_time = datetime.now()
            np.random.seed(42 + epoch + self.rank)
            pinn_loss_sum, batch_counter = self._train_epoch(optimizer, data_loader, epoch, writing_cycle,
                                                             track_gradient, activate_annealing, annealing_cycle)

            minimum_pinn_loss = self._log_training_progress(epoch, epochs, pinn_loss_sum, batch_counter,
                                                            epoch_start_time, writing_cycle, checkpoint_path,
                                                            save_model, pinn_path, hpm_path, logger, callbacks,
                                                            optimizer, minimum_pinn_loss)

        if lbfgs_finetuning:
            self._perform_lbfgs_finetuning(optimizer, data_loader, epochs, writing_cycle, save_model, pinn_path,
                                           hpm_path)

    def _train_epoch(self, optimizer, data_loader, epoch, writing_cycle, track_gradient, activate_annealing,
                     annealing_cycle):
        pinn_loss_sum = 0.0
        batch_counter = 0
        for idx, training_data in enumerate(data_loader):
            do_annealing = activate_annealing and not (epoch + 1) % annealing_cycle and idx == 0
            do_gradient_tracking = track_gradient and not (epoch + 1) % writing_cycle and idx == 0
            optimizer.zero_grad()
            pinn_loss = self.pinn_loss(training_data, do_gradient_tracking, do_annealing)
            pinn_loss.backward()
            optimizer.step()
            pinn_loss_sum += pinn_loss.item()
            batch_counter += 1
        return pinn_loss_sum, batch_counter

    def _log_training_progress(self, epoch, epochs, pinn_loss_sum, batch_counter, epoch_start_time, writing_cycle,
                               checkpoint_path, save_model, pinn_path, hpm_path, logger, callbacks, optimizer,
                               minimum_pinn_loss):
        if not self.rank:
            epoch_end_time = datetime.now()
            time_taken = (epoch_end_time - epoch_start_time).total_seconds()
            avg_pinn_loss = pinn_loss_sum / batch_counter
            all_losses = " | ".join(
                [f"{key} loss: {value / batch_counter:.6f}" for key, value in self.loss_log.items()])
            print(
                f"[{epoch_end_time}]:Epoch {epoch + 1:2d}/{epochs} | PINN Loss {avg_pinn_loss:.10f} | {all_losses} | Epoch Duration {time_taken:.5f}",
                flush=True)

            if logger is not None and not (epoch + 1) % writing_cycle:
                self._log_to_logger(logger, avg_pinn_loss, epoch)

            if callbacks is not None and not (epoch + 1) % writing_cycle:
                callbacks(epoch=epoch + 1)

            if avg_pinn_loss < minimum_pinn_loss and save_model:
                self.save_model(pinn_path, hpm_path)
                minimum_pinn_loss = avg_pinn_loss

            self._reset_loss_log()

            if not (epoch + 1) % writing_cycle and checkpoint_path is not None:
                self.write_checkpoint(checkpoint_path, epoch, False, minimum_pinn_loss, optimizer)

        return minimum_pinn_loss

    def _log_to_logger(self, logger, avg_pinn_loss, epoch):
        logger.log_scalar(scalar=avg_pinn_loss, name="Weighted PINN Loss", epoch=epoch + 1)
        logger.log_scalar(scalar=sum(self.loss_log.values()) / len(self.loss_log), name="Non-Weighted PINN Loss",
                          epoch=epoch + 1)

        for key, value in self.loss_log.items():
            logger.log_scalar(scalar=value / len(self.loss_log), name=key, epoch=epoch + 1)

        logger.log_scalar(scalar=self.initial_condition.weight, name=f"{self.initial_condition.name}_weight",
                          epoch=epoch + 1)
        logger.log_scalar(scalar=self.pde_loss.weight, name=f"{self.pde_loss.name}_weight", epoch=epoch + 1)

        if hasattr(self, 'loss_gradients_storage'):
            for key, gradients in self.loss_gradients_storage.items():
                logger.log_histogram(gradients.cpu(), f'gradients_{key}', epoch + 1)

        if not self.is_hpm:
            if isinstance(self.boundary_condition, list):
                for bc in self.boundary_condition:
                    logger.log_scalar(scalar=bc.weight, name=f"{bc.name}_weight", epoch=epoch + 1)
            else:
                logger.log_scalar(scalar=self.boundary_condition.weight, name=f"{self.boundary_condition.name}_weight",
                                  epoch=epoch + 1)

    def _reset_loss_log(self):
        for key in self.loss_log.keys():
            self.loss_log[key] = float(0)

    def _perform_lbfgs_finetuning(self, optimizer, data_loader, epochs, writing_cycle, save_model, pinn_path, hpm_path):
        def closure():
            optimizer.zero_grad()
            pinn_loss = self.pinn_loss(next(iter(data_loader)))
            pinn_loss.backward()
            return pinn_loss

        lbfgs_optim = torch.optim.LBFGS(self.model.parameters(), lr=0.9)
        lbfgs_optim.step(closure)
        pinn_loss = closure()
        print(f"After LBFGS-B: PINN Loss: {pinn_loss.item()} Epoch {epochs} from {epochs}")

        if pinn_loss < self.minimum_pinn_loss and not (epochs % writing_cycle) and save_model:
            self.save_model(pinn_path, hpm_path)

    def write_checkpoint(self, checkpoint_path, epoch, pretraining, minimum_pinn_loss, optimizer):
        checkpoint = {
            "epoch": epoch,
            "pretraining": pretraining,
            "minimum_pinn_loss": minimum_pinn_loss,
            "optimizer": optimizer.state_dict(),
            f"weight_{self.initial_condition.name}": self.initial_condition.weight,
            f"weight_{self.pde_loss.name}": self.pde_loss.weight,
            "pinn_model": self.model.state_dict()
        }

        if isinstance(self.boundary_condition, list):
            for bc in self.boundary_condition:
                checkpoint[f"weight_{bc.name}"] = bc.weight
        else:
            checkpoint[f"weight_{self.boundary_condition.name}"] = self.boundary_condition.weight

        if self.is_hpm:
            checkpoint["hpm_model"] = self.pde_loss.hpm_model.state_dict()

        torch.save(checkpoint, checkpoint_path)


    def take_snapshot(model, file_path, device, n_points):
        """
        Calculates a model output on a regular 3D grid and saves it as a VTK data.
        Args:
            model (nn.Module): a model predicting a scalar. It must have 'lb' and 'ub' attributes.
            file_path (str): a path of a file where VTK data will be saved.
            device (str): the device where the given model is located.
            n_points ([int,int,int]): number of points along each axis in the grid.        
        """
        assert len(model.lb) == 3  # Implemented only for 3D grid
        # evenly spaced numbers over a inteval specified by model.lb and model.ub
        x, y, z = [torch.linspace(model.lb[i], model.ub[i], n_points[i]) for i in range(len(model.lb))]
        x, y, z = torch.meshgrid(x, y, z)
        # create an input of shape [# points, 3]
        input = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], -1).view(-1, 3).to(device)
        # calculate the model output via minibatches of size 64
        output = torch.cat([model(input[k * 64:(k + 1) * 64]) for k in range((input.shape[0] // 64 + 1))], 0).view(-1)
        # convert all tensors to numpy arrays and save as VTK data
        grid = [x.numpy(), y.numpy(), z.numpy()]
        output = output.view(n_points).to('cpu').numpy()
        imageToVTK(file_path, grid, pointData={"model output": output})