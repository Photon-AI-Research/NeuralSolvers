import torch
import numpy as np
from argparse import ArgumentParser
import sys
import matplotlib.pyplot as plt
from torch.autograd import grad

sys.path.append('../..')  # PINNFramework etc.
import PINNFramework as pf
from IC_Dataset import ICDataset as ICDataset


def analyze(model, name, time, dataset, eval_bs=1048576):
    with torch.no_grad():
        num_batches = int(dataset.input_x.shape[0] / eval_bs)
        outputs = []
        for idx in range(num_batches):
            x = torch.tensor(dataset.input_x[eval_bs * idx: eval_bs * (idx + 1), :]).float().cuda()
            output = model(x).detach().cpu().numpy()
            outputs.append(output)

        pred = np.concatenate(outputs, axis=0)
        pred = pred.reshape(256, 2048, 256)

        fig1 = plt.figure()
        slc = pred[:, :, 120]
        plt.imshow(slc, cmap='jet', aspect='auto')
        plt.colorbar()
        plt.xlabel("z")
        plt.ylabel("y")

        fig2 = plt.figure()
        slc = pred[:, 800, :]
        plt.imshow(slc, cmap='jet', aspect='auto')
        plt.colorbar()
        plt.xlabel("z")
        plt.ylabel("x")

        fig3 = plt.figure()
        slc = pred[120, :, :]
        plt.imshow(slc, cmap='jet', aspect='auto')
        plt.colorbar()
        plt.xlabel("y")
        plt.ylabel("x")
        np.save("pred_"+name+"_"+str(time),pred)
        np.save("gt_"+ str(time),dataset.e_field)
        np.save("training_x_" + str(time), dataset.input_x)
        np.save("training_y_" + str(time), dataset.e_field)
        del pred  # clear memory

        fig1.savefig("zy_{}_{}.png".format(name, time))
        fig2.savefig("zx_{}_{}.png".format(name, time))
        fig3.savefig("yx_{}_{}.png".format(name, time))

def wave_eq(x, u):

    grads = torch.ones(u.shape, device=u.device)  # move to the same device as prediction

    grad_u = grad(u, x, create_graph=True, grad_outputs=grads)[0]  # (z, y, x, t)

    u_z = grad_u[:, 0]
    u_y = grad_u[:, 1]
    u_x = grad_u[:, 2]
    u_t = grad_u[:, 3]

    grads = torch.ones(u_z.shape, device=u_z.device) # update for shapes

    # calculate second order derivatives
    u_zz = grad(u_z, x, create_graph=True, grad_outputs=grads)[0][:, 0]  # (z, y, x, t)
    u_yy = grad(u_y, x, create_graph=True, grad_outputs=grads)[0][:, 1]
    u_xx = grad(u_x, x, create_graph=True, grad_outputs=grads)[0][:, 2]
    u_tt = grad(u_t, x, create_graph=True, grad_outputs=grads)[0][:, 3]
    f_u = u_tt - (u_zz + u_yy + u_xx)
    return f_u, u_tt, u_zz, u_yy, u_xx, u_y, u_t


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
    print("lb:",dataset_2000.lb)
    print("ub:",dataset_2000.ub)
    dataset_2100 = ICDataset(args.path, 2100, 10000, 0, 2100, False)
    #dataset_2200 = ICDataset(args.path, 2200, 10000, 0, 2200, False)

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
    #analyze(model, args.name, 2000, dataset_2000)
    #analyze(model, args.name, 2100, dataset_2100)
    input_x = dataset_2000.input_x
    input_x = input_x.reshape(256, 2048, 256,4)
    input_x = input_x[128:133, : , 128:133, :]
    for i in range(2000, 2200, 10):
        x = torch.Tensor(input_x.reshape(-1, 4))
        x[:, 3] = i
        x = x.float().cuda()
        x.requires_grad = True
        u = model(x)
        f_u, u_tt, u_zz, u_yy, u_xx, u_y, u_t = wave_eq(x, u)
        np.save("interpolation/prediction{}".format(i), u.detach().cpu().numpy())
        np.save("interpolation/f_u_{}".format(i), f_u.detach().cpu().numpy())
        np.save("interpolation/u_tt_{}".format(i), u_tt.detach().cpu().numpy())
        np.save("interpolation/u_zz_{}".format(i), u_zz.detach().cpu().numpy())
        np.save("interpolation/u_yy_{}".format(i), u_yy.detach().cpu().numpy())
        np.save("interpolation/u_xx_{}".format(i), u_xx.detach().cpu().numpy())
        np.save("interpolation/u_y_{}".format(i), u_y.detach().cpu().numpy())
        np.save("interpolation/u_t_{}".format(i), u_t.detach().cpu().numpy())





