from NeuralSolvers.models import FingerMoE
import numpy as np
import torch

if __name__ == "__main__":
    lb = np.array([0, 0, 0, 0])
    ub = np.array([1, 1, 1, 1])

    # finger MoE gpu
    moe = FingerMoE(4, 3, 5, 100, 2, lb, ub)
    moe.cuda()
    x_gpu = torch.randn(3, 4).cuda()
    y_gpu = moe(x_gpu)
    print(y_gpu)

    # finger MoE cpu
    moe.cpu()
    x_cpu = torch.randn(3, 4)
    y_cpu = moe(x_cpu)
    print(y_cpu)

    # non linear gating test
    moe =  FingerMoE(4, 3, 5, 100, 2, lb, ub, non_linear=True)
    moe.cuda()
    x_gpu = torch.randn(3, 4).cuda()
    y_gpu = moe(x_gpu)
    print(y_gpu)
    moe.cpu()
    x_cpu = torch.randn(3, 4)
    y_cpu = moe(x_cpu)
    print(y_cpu)

