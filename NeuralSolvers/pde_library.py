import torch
from torch.autograd import grad
from torch import ones, stack


def burgers1D(params):
    viscosity = params["viscosity"]

    def pde(x, u):
        grads = ones(u.shape, device=u.device)
        grad_u = grad(u, x, create_graph=True, grad_outputs=grads)[0]
        u_x, u_t = grad_u[:, 0], grad_u[:, 1]

        grads_x = ones(u_x.shape, device=u.device)
        u_xx = grad(grad_u[:, 0], x, create_graph=True, grad_outputs=grads_x)[0][:, 0]

        return u_t + u * u_x - viscosity * u_xx

    return pde

def wave1D(params):
    wave_speed = params["wave_speed"]

    def pde(x, u):
        grads = ones(u.shape, device=u.device)
        grad_u = grad(u, x, create_graph=True, grad_outputs=grads)[0]
        u_x, u_t = grad_u[:, 0], grad_u[:, 1]

        grads_x = ones(u_x.shape, device=u.device)
        u_xx = grad(grad_u[:, 0], x, create_graph=True, grad_outputs=grads_x)[0][:, 0]

        grads_t = ones(u_t.shape, device=u.device)
        u_tt = grad(grad_u[:, 1], x, create_graph=True, grad_outputs=grads_t)[0][:, 1]

        return u_tt - wave_speed**2 * u_xx

    return pde

def schrodinger1D(params):
    def pde(x, u):
        u_real, u_imag = u[:, 0], u[:, 1]

        grads = ones(u_real.shape, device=u.device)
        grad_u_real = grad(u_real, x, create_graph=True, grad_outputs=grads)[0]
        grad_u_imag = grad(u_imag, x, create_graph=True, grad_outputs=grads)[0]

        u_real_x, u_real_t = grad_u_real[:, 0], grad_u_real[:, 1]
        u_imag_x, u_imag_t = grad_u_imag[:, 0], grad_u_imag[:, 1]

        u_real_xx = grad(u_real_x, x, create_graph=True, grad_outputs=grads)[0][:, 0]
        u_imag_xx = grad(u_imag_x, x, create_graph=True, grad_outputs=grads)[0][:, 0]

        f_real = u_real_t + 0.5 * u_imag_xx + (u_real ** 2 + u_imag ** 2) * u_imag
        f_imag = u_imag_t - 0.5 * u_real_xx - (u_real ** 2 + u_imag ** 2) * u_real

        return stack([f_real, f_imag], 1)

    return pde

def heat1D(params):
    diffusivity = params["diffusivity"]
    def pde(x, u):
        grads = ones(u.shape, device=u.device)
        grad_u = grad(u, x, create_graph=True, grad_outputs=grads)[0]
        u_x, u_t = grad_u[:, 0], grad_u[:, 1]

        grads = ones(u_x.shape, device=u.device)
        u_xx = grad(u_x, x, create_graph=True, grad_outputs=grads)[0][:, 0]

        u_x, u_t, u_xx = [tensor.reshape(-1, 1) for tensor in (u_x, u_t, u_xx)]

        return u_t - u_xx

    return pde