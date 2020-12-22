from LossTerm import LossTerm
from torch.autograd import grad
from torch import ones


class BoundaryCondition(LossTerm):
    def __init__(self, norm='L2'):
        super(BoundaryCondition).__init__(norm)


class DirichletBC(BoundaryCondition):
    """
    Dirichlet boundary conditions: y(x) = func(x).
    """

    def __init__(self, func, norm='L2'):
        super(DirichletBC).__init__(norm)
        self.func = func

    def __call__(self, x, model):
        prediction = model(x)  # is equal to y
        return self.norm(prediction, self.func(x))


class NeumannBC(BoundaryCondition):
    """
    Neumann boundary conditions: dy/dn(x) = func(x).
    """

    def __init__(self, func, dimension, norm='L2'):
        super(NeumannBC).__init__(norm)
        self.func = func
        self.dimension = dimension

    def __call__(self, x, model):
        grads = ones(x.shape, device=x.device)
        y = model(x)
        grad_y = grad(y, x, create_graph=True, grad_outputs=grads)[0]
        y_dn = grad_y[:, self.dimension]
        return self.norm(y_dn, self.func(x))


class RobinBC(BoundaryCondition):
    """
    Robin boundary conditions: dy/dn(x) = func(x, y).
    """

    def __init__(self, func, dimension, norm='L2'):
        super(NeumannBC).__init__(norm)
        self.func = func
        self.dimension = dimension

    def __call__(self, x, model, y):
        grads = ones(x.shape, device=x.device)
        y = model(x)
        grad_y = grad(y, x, create_graph=True, grad_outputs=grads)[0]
        y_dn = grad_y[:, self.dimension]
        return self.norm(y_dn, self.func(x, y))


class PeriodicBC(BoundaryCondition):
    """
    Periodic boundary condition
    """

    def __init__(self, dimension, degree=0, norm='L2'):
        super(PeriodicBC).__init__(norm)
        self.dimension = dimension
        self.degree = degree

    def __call__(self, x_lb, x_ub, model):
        y_lb = model(x_lb)
        y_ub = model(x_ub)
        grads = ones(x_lb.shape, device=x_lb.device)
        if self.degree == 0:
            return self.norm(y_lb, y_ub)
        elif self.degree == 1:
            y_lb_grad = grad(y_lb, x_lb, create_graph=True, grad_outputs=grads)[0]
            y_ub_grad = grad(y_lb, x_ub, create_graph=True, grad_outputs=grads)[0]
            y_lb_dn = y_lb_grad[:, self.dimension]
            y_ub_dn = y_ub_grad[:, self.dimension]
            return self.norm(y_lb_dn, y_ub_dn)
        else:
            raise NotImplementedError("Periodic Boundary Condition for a higher degree than one is not supported")
