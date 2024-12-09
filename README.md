<p float="left" align="center"> 
<img src="images/cropped_logo.png" width="20%" />
<img src="images/1D_Schroedinger_training.gif" width="35%"/>
</p>


# Introduction
Neural Solvers are neural network based solver for partial differential equations and inverse problems. 
The framework implements the physics-informed neural network approach on scale. Physics informed neural networks
allow strong scaling by design. Therefore, we have developed a framework that uses data parallelism to accelerate the training of 
physics informed neural networks significantly. To implement data parallelism, we use the <a href="https://github.com/horovod/horovod">Horovod</a> framework, which provides near-ideal speedup on multi-GPU regimes.  

<figure align="center">
<img src="images/scalability.png" width="50%" />
</figure>

More details about our framework you can find in our recent publication: 
```
P. Stiller, F. Bethke, M. Böhme, R. Pausch, S. Torge, A. Debus, J. Vorberger, M.Bussmann, N. Hoffmann: 
Large-scale Neural Solvers for Partial Differential Equations (2020).

```

## Implemented Approaches:

- P. Stiller, F. Bethke, M. Böhme, R. Pausch, S. Torge, A. Debus, J. Vorberger, M.Bussmann, N. Hoffmann: 
Large-scale Neural Solvers for Partial Differential Equations (2020).


- Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis.
Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations.(2017).

- Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. 
Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations.(2017).

- Suryanarayana Maddu, Dominik Sturm, Christian L. Müller and Ivo F. Sbalzarini (2021):
Inverse Dirichlet Weighting Enables Reliable Training of Physics Informed Neural Networks


- Sifan Wang, Yujun Teng, Paris Perdikaris (2020)
Understanding and mitigating gradient pathologies in physics-informed neural networks

- Mohammad Amin Nabian, Rini Jasmine Gladstone, Hadi Meidani (2021)
efficient training of physics-informed neural networks via importance sampling


## Requirements

### Libaries
```
cuda 10.2 # if gpu support is needed
python/3.6.5
gcc/5.5.0
openmpi/3.1.2
```

### Python requirements
```
torch>=1.7.1 
h5py>=2.10.0
numpy>=1.19.0
Pillow>=7.2.0
matplotlib>=3.3.3
scipy>=1.6.1
pyDOE>=0.3.8
```

## Usage of Interface

At the beginning you have to implement the datasets following the torch.utils.Dataset interface

```python
from torch.utils.data import Dataset
import NeuralSolvers as nsolv


class BoundaryConditionDataset(Dataset):

    def __init__(self, nb, lb, ub):
        """
        Constructor of the initial condition dataset
		"""

    def __getitem__(self, idx):
        """
        Returns data for initial state
        """

    def __len__(self):
        """
		Length of the dataset
        """


class InitialConditionDataset(Dataset):

    def __init__(self, **kwargs):
        """
        Constructor of the boundary condition dataset

        """

    def __len__(self):
        """
		Length of the dataset
        """

    def __getitem__(self, idx):
        """
        Returns item at given index
        """


class PDEDataset(Dataset):
    def __init__(self, nf, lb, ub):
        """
		Constructor of the PDE dataset
		"""

    def __len__(self):
        """
		Length of the dataset
        """

    def __getitem__(self, idx):
        """
        Returns item at given index
        """

```

In the main function you can create the loss-terms and the corresponding datasets. 
And define the pde function f which is the residual of the pde given residual points and model predictions u.
For the boundary conditions: neumann, robin, dirchlet and periodic boundary condititions are supported. 

```python

if __name__ == main :

    # initial condition
    ic_dataset = InitialConditionDataset(...)
    initial_condition = nsolv.InitialCondition(dataset=ic_dataset)
    # boundary conditions
    bc_dataset = BoundaryConditionDataset(...)
    periodic_bc_u = nsolv.PeriodicBC(...)
    periodic_bc_v = nsolv.PeriodicBC(...)
    periodic_bc_u_x = nsolv.RobinBC(...)
    periodic_bc_v_x = nsolv.NeumannBC(...)
    # PDE 
	pde_dataset = PDEDataset(...)


    def f(x, u):
		"""
		
		Defines the residual of the pde f(x,u)=0
		
		"""


    pde_loss = nsolv.PDELoss(dataset=pde_dataset, func=f)

```

Finally you can create a model which is the surrogate for the PDE and create the PINN enviorment which helps you to train the surrogate.
```python
model = nsolv.models.MLP(input_size=2, output_size=2, hidden_size=100, num_hidden=4) # creating a model. For example a mlp
pinn = nsolv.PINN(model, input_size=2, output_size=2 ,pde_loss = pde_loss, initial_condition=initial_condition, boundary_condition = [...], use_gpu=True)

pinn.fit(50000, 'Adam', 1e-3)
```

## Deep HPM support 

Instead of a PDE loss you can use a HPM model. The HPM model needs a function derivatives that calculates the needed derivatives, while the last returned derivative is the time_derivative.
You can use the HPM loss a follows. 

```python

def derivatives(x,u):
	"""
	Returns the derivatives
	
	Args: 
		x (torch.Tensor) : residual points
		u (torch.Tensor) : predictions of the pde model
	"""
	pass
	
hpm_loss = nsolv.HPMLoss(pde_dataset,derivatives,hpm_model)
#HPM has no boundary conditions in general
pinn = nsolv.PINN(model, input_size=2, output_size=2 ,pde_loss = hpm_loss, initial_condition=initial_condition, boundary_condition = [], use_gpu=True)

```
## Horovod Support 
You can activate horovod support by setting the `use_horovod` flag in the constructor of the pinn
```python
pinn = nsolv.PINN(model, input_size=2, output_size=2 ,pde_loss = pde_loss, initial_condition=initial_condition, boundary_condition = [...], use_gpu=True, use_horovod=True)
Keep in mind that the lbfgs-optimizer and the lbgfgs-finetuning is not supported with horovod activated. Another restriction is that the length or your dataset should not be smaller than the number of used GPUs for horovod.
```
## Wandb support 
Activate wandb-logging by creating an instance of a wandb logging. Its important that you have wandb installed. 
Look here for installing wandb: https://docs.wandb.ai/quickstart
```python
logger = nsolv.WandbLogger(project, args) # create logger instance
pinn.fit(epochs=5000,logger=logger) # add logger to the fit method
```
## Tensorboard support 
Activate tensorboard-logging by creating an event file with tensorboardX. Its important that you have tensorboardX installed. 
```python
logger = nsolv.TensorBoardLogger(log_directory) # create logger instance
pinn.fit(epochs=5000,logger=logger) # add logger to the fit method
```
 


## Developers

### Scientific Supervision
Nico Hoffmann (HZDR)
### Core Developers 
Patrick Stiller (HZDR) <br/>
Maksim Zhdanov (HZDR)<br/>
Jeyhun Rustamov (HZDR) <br/>
Raj Dhansukhbhai Sutariya (HZDR) <br/>
