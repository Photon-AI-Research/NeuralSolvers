from PINNFramework.models import MoE
import numpy as np
import torch

if __name__ == "__main__":
    lb = np.array([0, 0, 0])
    ub = np.array([1, 1, 1])

    # linear gating test
    moe = MoE(3, 3, 5, 100, 2, lb, ub)
    moe.cuda()
    x_gpu = torch.randn(3, 3).cuda()
    y_gpu = moe(x_gpu)
    print(y_gpu)
    moe.cpu()
    x_cpu = torch.randn(3, 3)
    y_cpu = moe(x_cpu)
    print(y_cpu)

    # non linear gating test
    moe = MoE(3, 3, 5, 100, 2, lb, ub, non_linear=True)
    moe.cuda()
    x_gpu = torch.randn(3, 3).cuda()
    y_gpu = moe(x_gpu)
    print(y_gpu)
    moe.cpu()
    x_cpu = torch.randn(3, 3)
    y_cpu = moe(x_cpu)
    print(y_cpu)
