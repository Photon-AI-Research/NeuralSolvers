import torch
import time
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--distributed", dest="distributed", type=int, default=0)
args = parser.parse_args()

if args.distributed:
    from PINNFramework.models.distributed_moe import MoE
else:
    from PINNFramework.models.moe import MoE

if __name__ == "__main__":
    model = MoE(3, 3, 10, 300, 5, lb=[0, 0, 0], ub=[1, 1, 1], device='cuda:0', k=1).eval()
    times = []
    for i in range(100000, 352000, 2000):
        x = torch.randn((i, 3)).cuda()
        torch.cuda.synchronize()
        begin_time = time.time()
        model.forward(x, train=False)
        torch.cuda.synchronize()
        end_time = time.time()
        run_time = (end_time - begin_time)
        print("For {} samples: {} sec".format(i, run_time))
        times.append([i,
                      run_time])
        del x
        torch.cuda.empty_cache()
    if args.distributed:
        np.save("dist_run_time", times)
    else:
        np.save("non_dist_run_time", times)


    """
    times = np.array(times)
    plt.scatter(times[1:, 0],times[1:, 1], s=1)
    plt.xlabel("Number of input samples")
    plt.ylabel("Inference time")
    plt.show()
    """

