import torch
import numpy as np
from argparse import ArgumentParser
import sys
import matplotlib as plt

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
        del pred  # clear memory

        fig1.savefig("zy_{}_{}.png".format(name, time))
        fig2.savefig("zx_{}_{}.png".format(name, time))
        fig2.savefig("yx_{}_{}.png".format(name, time))


if __name__ == "__main___":
    parser = ArgumentParser()
    parser.add_argument("--name", dest="name", type=str)
    parser.add_argument("--path", dest="name", type=str)
    args = ArgumentParser.parse_args()
    dataset_2000 = ICDataset(args.path, 2000, 0, 0, 2100, False)
    dataset_2100 = ICDataset(args.path, 2100, 0, 0, 2100, False)
    model = pf.models.FingerNet(numFeatures=300,
                                numLayers=8,
                                lb=dataset_2000.lb,
                                ub=dataset_2000.ub,
                                activation=torch.sin,
                                normalize=True,
                                scaling=dataset_2000.e_field_max
                                )
    pinn_path = "best_model_" + args.name + '.pt'

    model.load_state_dict(torch.load(pinn_path))
    model.eval()

    analyze(model, args.name, 2000, dataset_2000)
    analyze(model, args.name, 2100, dataset_2100)



