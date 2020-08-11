# NeuralSolvers
Neural network based solvers for partial differential equations.

## API description

![Test](./images/API_PINN.png)

## Requirements

### Libaries
```
cuda 10.2
python/3.6.5
gcc/5.5.0
openmpi/3.1.2
```

### Python requirements
```
torch==1.5.1
h5py==2.10.0
horovod==0.19.5
tensorboard==2.1.0
tensorboardX==2.0
numpy==1.19.0
Pillow==7.2.0
matplotlib==3.1.3
scipy==1.4.1
```

## Usage of Interface
You can use the Interface to implement a new PINN module.  You must implement the PINN loss so that ```forward()``` has to be called **only once**.

```python
import PINN.Interface as Interface

class ModifiedPINN(Interface):

	def forward(self):
		""" Implement the forward pass """
		pass
	
	def derivatives(self):
		""" Calculation of derivatives """
		
	def intitial_loss(self):
		""" Implement the initial loss """
		pass
		
	def pde_loss(self):
		""" Implement the pde loss """
		pass
	
	def boundardy_loss(self):
		""" Implement the boundary loss """
		pass
	
	def pinn_loss(self):
		""" Implement the PINN loss """
		pass
		
```