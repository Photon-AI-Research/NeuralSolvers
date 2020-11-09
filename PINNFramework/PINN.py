import torch
import torch.nn as nn


class PINN(nn.Module):
    
    def __init__(self, model: torch.nn.Module, input_dimension: int , output_dimension: int, pde_loss: PDELoss, initial_condition :InitialCondition, boundary_condition):
        r"""
        Initializes an physics-informed neural network(PINN). A PINN consists of a model which represents the solution of the underlying partial differential equation(PDE) u, 
        three loss terms representing initial (IC) and boundary condtion(BC) and the PDE and a dataset which represents the bounded domain U.

        Args: 
            model : is the model which is trained to represent the underlying PDE
            input_dimension : represents the dimension of the input vector x
            output_dimension : represents the dimension of the solution u
            pde_loss: Instance of the PDELoss class. Represents the underlying PDE
            initial_condition: Instance of the InitialCondition class. Represents the initial condition
            boundary condition (BoundaryCondition, list): Instance of the BoundaryCondition class or a list of instances of the BoundaryCondition class

        """

        super(PINN, self).__init__()
        # checking if the model is a torch module more model checking should be possible
        if isinstance(model, nn.Module):
            self.model = model
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
        else: 
            raise TypeError("PDE loss has to be an instance of a PDELoss class")
            
        if isinstance(initial_condition, InitialCondition):
                self.initial_condition = initial_condition
        else: 
            raise TypeError("Initial condition has to be an instance of the InitialCondition class")

        if type(boundary_conditions) is list: 
            for bc in boundary_conditions:
                if not isinstance(bc,BoundaryCondition):
                    raise TypeError("Boundary Condition has to be an instance of the BoundaryCondition class ")
            self.boundary_condition = boundary_condition
        else:
            if isinstance(boundary_condition,BoundaryCondition):
                self.boundary_condition = boundary_condition
            else:
                raise TypeError("Boundary Condation has to be an instance of the BoundaryCondition class or a list of instances of the BoundaryCondition class")

        # TODO creating dataset from loss function 
    

    
    def forward(self,x):
        """
        Predicting the solution at given position x
        """
        return self.model(x)
    
    def pinn_loss(self, x, y):
        pde_loss = self.self.pde_loss(x["pde"],model)
        initial_loss = self.initial_loss(x["pde"],y["pde"],model)
        if type(self.boundary_loss) list:
            boundary_loss = 0 
            for b in self.boundary_condition:
                boundary_loss = boundary_loss + boundary_loss(x[b.name],y[b.name], model)
        else:
            boundary_loss = self.boundary_condition(x[self.boundary_condition.name], y[self.boundary_condition.name], model)
        return pde_loss, initial_loss, boundary_loss


    

    
