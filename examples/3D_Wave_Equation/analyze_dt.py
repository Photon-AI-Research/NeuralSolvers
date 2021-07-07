import torch
import numpy as np
from argparse import ArgumentParser
import sys
sys.path.append('../..')  # PINNFramework etc.
import PINNFramework as pf
from IC_Dataset import ICDataset as ICDataset
from torch.autograd import grad


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", dest="name", type=str)
    parser.add_argument("--path", dest="path", type=str)
    args = parser.parse_args()
    dataset_2000 = ICDataset(args.path, 2000, 5000, 0, 2100, False)
    print("cell_depth:", dataset_2000.cell_depth)
    print("cell_height:", dataset_2000.cell_height)
    print("cell_width:", dataset_2000.cell_width)
    print("scaling:", dataset_2000.e_field_max)
    print("lb:", dataset_2000.lb)
    print("ub:", dataset_2000.ub)
    # dataset_2100 = ICDataset(args.path, 2100, 10000, 0, 2100, False)
    # dataset_2200 = ICDataset(args.path, 2200, 10000, 0, 2200, False)

    model = pf.models.FingerNet(numFeatures=300,
                                numLayers=8,
                                lb=dataset_2000.lb,
                                ub=dataset_2000.ub,
                                activation=torch.sin,
                                normalize=True,
                                scaling=dataset_2000.e_field_max
                                )
    model.cuda()
    pinn_path = "best_model_" + args.name + '.pt'

    model.load_state_dict(torch.load(pinn_path))
    model.eval()

    dt = []
    batch_size = int(2**14)

    for i in range((256 * 256 * 2048) // batch_size):
        x = torch.tensor(dataset_2000.input_2000[i*batch_size: (i+1)*batch_size, :], device='cuda:0')
        u = model(x)
        grads = torch.ones(u.shape, device=u.device)  # move to the same device as prediction
        grad_u = grad(u, x, create_graph=True, grad_outputs=grads)[0]  # (z, y, x, t)
        u_t = grad_u[:, 3]
        u_t = u_t.reshape(-1, 1)
        dt.append(u_t.detach().cpu().numpy())

    np.save('dt', np.concatenate(dt))

