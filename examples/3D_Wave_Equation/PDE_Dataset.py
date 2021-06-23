import numpy as np
import openpmd_api as io
from torch import Tensor
from pyDOE import lhs
from torch.utils.data import Dataset
import openpmd_api as io

class PDEDataset(Dataset):
    def __init__(self, path, nf, batch_size, iteration, tmax):
        """
        Constructor of the PDE Dataset

        Args:
            lb: defines the lower bound of the spatial temporal domain
            ub: defines the uppper bound of the spatial temporal domain
            nf: defines the number of residual points used in total
            batch_size: defines the number of residual points yielded in a batch
        """


        self.batch_size = batch_size
        self.nf = nf

        # loading mesh specifications
        series = io.Series(path, io.Access_Type.read_only)
        it = series.iterations[iteration]
        self.cell_depth = it.get_attribute('cell_depth')
        self.cell_height = it.get_attribute('cell_height')
        self.cell_width = it.get_attribute('cell_width')

        e_x = it.meshes["E"]["x"].load_chunk()
        series.flush()

        field_shape = e_x.shape  # (z, y, x)
        z_length = field_shape[0]
        y_length = field_shape[1]
        x_length = field_shape[2]

        z = np.arange(0, z_length, 8) * self.cell_depth  # reduce sampling in z-direction
        y = np.arange(0, y_length) * self.cell_height
        x = np.arange(0, x_length, 8) * self.cell_width  # reduce sampling in x-direction
        t = np.arange(iteration, tmax+1)

        Z, Y, X, T = np.meshgrid(z, y, x, t, indexing='ij')
        z = Z.reshape(-1, 1)
        x = X.reshape(-1, 1)
        y = Y.reshape(-1, 1)
        t = T.reshape(-1, 1)
        self.samples = np.concatenate([z, y, x, t], axis=1)
        rand_idx = np.random.choice(self.samples.shape[0], self.nf, replace=False)
        self.xf = self.samples[rand_idx, :]


    def __len__(self):
        """
        Returns the number of batches returned by the dataset
        """
        return self.nf // self.batch_size

    def __getitem__(self, item):
        """
        Yields the batches of xf
        """
        return Tensor(self.xf[item * self.batch_size: (item + 1) * self.batch_size]).float()

