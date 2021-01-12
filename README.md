# NeuralSolvers
Neural network based solvers for partial differential equations.


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
torch==1.7.1 
h5py==2.10.0
numpy==1.19.0
Pillow==7.2.0
matplotlib==3.3.3
scipy==1.6.1
pyDOE==0.3.8
```

## Usage of Interface

At the beginning you have to implement the datasets following the torch.utils.Dataset interface
```python
import torch.utils.Dataset as Dataset
import PINN.Interface as Interface

sys.path.append(PATH_TO_PINN_FRAMEWORK)  # adding the pinn framework to your path
import PINNFramework as pf

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
    initial_condition = pf.InitialCondition(dataset=ic_dataset)
    # boundary conditions
    bc_dataset = BoundaryConditionDataset(...)
    periodic_bc_u = pf.PeriodicBC(...)
    periodic_bc_v = pf.PeriodicBC(...)
    periodic_bc_u_x = pf.RobinBC(...)
    periodic_bc_v_x = pf.NeumannBC(...)
    # PDE 
	pde_dataset = PDEDataset(...)


    def f(x, u):
		"""
		
		Defines the residual of the pde f(x,u)=0
		
		"""


    pde_loss = pf.PDELoss(dataset=pde_dataset, func=f)

```

Finally you can create a model which is the surrogate for the PDE and create the PINN enviorment which helps you to train the surrogate.
```python
model = pf.models.MLP(input_size=2, output_size=2, hidden_size=100, num_hidden=4) # creating a model. For example a mlp
pinn = pf.PINN(model, input_size=2, output_size=2 ,pde_loss = pde_loss, initial_condition=initial_condition, boundary_condition = [...], use_gpu=True)

pinn.fit(50000, 'Adam', 1e-3)
```