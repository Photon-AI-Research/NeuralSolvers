import torch
from torch.autograd import grad
from torch import ones


def burgers1D(x, u, params):
    viscosity = params["viscosity"]
    grads = ones(u.shape, device=u.device)
    grad_u = grad(u, x, create_graph=True, grad_outputs=grads)[0]
    u_x, u_t = grad_u[:, 0], grad_u[:, 1]
    u_xx = grad(grad_u[:, 0], x, create_graph=True, grad_outputs=grads)[0][:, 0]
    return u_t + u * u_x - viscosity * u_xx

def wave1D(x, u, params):
    wave_speed = params["wave_speed"]
    grads = ones(u.shape, device=u.device)
    grad_u = grad(u, x, create_graph=True, grad_outputs=grads)[0]
    u_x, u_t = grad_u[:, 0], grad_u[:, 1]
    u_xx = grad(grad_u[:, 0], x, create_graph=True, grad_outputs=grads)[0][:, 0]
    u_tt = grad(grad_u[:, 1], x, create_graph=True, grad_outputs=grads)[0][:, 1]
    return u_tt - wave_speed**2 * u_xx

def schrodinger1D(x, u, params):
    potential = params["potential"]
    grads = ones(u.shape, device=u.device)
    grad_u = grad(u, x, create_graph=True, grad_outputs=grads)[0]
    u_xx = grad(grad_u[:, 0], x, create_graph=True, grad_outputs=grads)[0][:, 0]
    return torch.imag(grad_u[:, 1]) - u_xx + potential * u