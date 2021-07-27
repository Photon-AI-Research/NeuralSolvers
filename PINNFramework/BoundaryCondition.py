from .LossTerm import LossTerm
from torch.autograd import grad
from torch import ones


class BoundaryCondition(LossTerm):
    def __init__(self, dataset, name, norm='L2', weight=1.):
        super(BoundaryCondition, self).__init__(dataset, name, norm, weight)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("The call function of the Boundary Condition has to be implemented")


class DirichletBC(BoundaryCondition):
    """
    Dirichlet boundary conditions: y(x) = func(x).
    """

    def __init__(self, func, dataset, name, norm='L2',weight=1.):
        super(DirichletBC, self).__init__(dataset, name, norm, weight)
        self.func = func

    def __call__(self, x, model):
        prediction = model(x)  # is equal to y
        return self.weight * self.norm(prediction, self.func(x))


class NeumannBC(BoundaryCondition):
    """
    Neumann boundary conditions: dy/dn(x) = func(x).
    With dy/dn(x) = <∇y,n>
    """

    def __init__(self, func, dataset, normal_vector, begin, end, output_dimension, name, norm='L2', weight=1.):
        """
        Args:
            func: scalar but vectorized function f(x)
            normal_vector: normal vector for the face
            name: identifier of the boundary condition
            weight: weighting of the boundary condition
            begin: defines the begin of spatial variables in x
            end: defines the end of the spatial domain in x
            output_dimension defines on which dimension of the output the boundary condition performed
        """
        super(NeumannBC, self).__init__(dataset, name, norm, weight)
        self.func = func
        self.normal_vector = normal_vector
        self.begin = begin
        self.end = end
        self.output_dimension = output_dimension

    def __call__(self, x, model):
        x.requires_grad = True
        y = model(x)
        y = y[:, self.output_dimension]
        grads = ones(y.shape, device=y.device)
        grad_y = grad(y, x, create_graph=True, grad_outputs=grads)[0]
        grad_y = grad_y[:,self.begin:self.end]
        self.normal_vector.to(y.device)  # move normal vector to the correct device
        y_dn = grad_y @ self.normal_vector
        return self.weight * self.norm(y_dn, self.func(x))


class RobinBC(BoundaryCondition):
    """
    Robin boundary conditions: dy/dn(x) = func(x, y).
    """

    def __init__(self, func, dataset, normal_vector, begin, end, output_dimension, name, norm='L2', weight=1.):
        """
            Args:
                func: scalar but vectorized function f(x,y)
                normal_vector: normal vector for the face
                name: identifier of the boundary condition
                weight: weighting of the boundary condition
                begin: defines the begin of spatial variables in x
                end: defines the end of the spatial domain in x
                output_dimension defines on which dimension of the output the boundary condition performed
        """

        super(RobinBC, self).__init__(dataset, name, norm, weight)
        self.func = func
        self.begin = begin
        self.end = end
        self.normal_vector = normal_vector
        self.output_dimension = output_dimension

    def __call__(self, x, y, model):
        x.requires_grad = True
        y = model(x)
        y = y[:, self.output_dimension]
        grads = ones(y.shape, device=y.device)
        grad_y = grad(y, x, create_graph=True, grad_outputs=grads)[0]
        grad_y = grad_y[:, self.begin:self.end]
        self.normal_vector.to(y.device)  # move normal vector to the correct device
        y_dn = grad_y @ self.normal_vector
        return self.weight * self.norm(y_dn, self.func(x, y))


class PeriodicBC(BoundaryCondition):
    """
    Periodic boundary condition
    """

    def __init__(self, dataset, output_dimension, name, degree=None, input_dimension=None,  norm='L2', weight=1.):
        super(PeriodicBC, self).__init__(dataset, name, norm, weight)
        if degree is not None and input_dimension is None:
            raise ValueError("If the degree of the boundary condition is defined the input dimension for the "
                             "derivative has to be defined too ")
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.degree = degree

    def __call__(self, x_lb, x_ub, model):
        x_lb.requires_grad = True
        x_ub.requires_grad = True
        y_lb = model(x_lb)[:, self.output_dimension]
        y_ub = model(x_ub)[:, self.output_dimension]
        grads = ones(y_lb.shape, device=y_ub.device)
        if self.degree is None:
            return self.weight * self.norm(y_lb, y_ub)
        elif self.degree == 1:
            y_lb_grad = grad(y_lb, x_lb, create_graph=True, grad_outputs=grads)[0]
            y_ub_grad = grad(y_ub, x_ub, create_graph=True, grad_outputs=grads)[0]
            y_lb_dn = y_lb_grad[:, self.input_dimension]
            y_ub_dn = y_ub_grad[:, self.input_dimension]
            return self.weight * self.norm(y_lb_dn, y_ub_dn)

        else:
            raise NotImplementedError("Periodic Boundary Condition for a higher degree than one is not supported")


class TimeDerivativeBC(BoundaryCondition):
    """
    For hyperbolic systems it may be needed to initialize the time derivative. This boundary condition intializes
    the time derivative in a data driven way.
    """
    def __init__(self, dataset, name, norm='L2', weight=1):
        super(TimeDerivativeBC, self).__init__(dataset, name, norm, weight)

    def __call__(self, x, dt_y, model):
        x.requires_grad = True
        pred = model(x)
        grads = ones(pred.shape, device=pred.device)
        pred_dt = grad(pred, x, create_graph=True, grad_outputs=grads)[0][:, -1]
        pred_dt = pred_dt.reshape(-1,1)
        return self.weight * self.norm(pred_dt, dt_y)