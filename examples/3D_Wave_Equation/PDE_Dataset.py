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
        self.tmax = tmax
        self.iteration = iteration

        e_x = it.meshes["E"]["x"].load_chunk()
        series.flush()

    def __len__(self):
        """
        Returns the number of batches returned by the dataset
        """
        return self.nf // self.batch_size

    def __getitem__(self, item):
        """
        Yields the batches of xf
        """
        z = np.random.normal(128, 10, size=self.batch_size).reshape(-1, 1) * self.cell_depth
        x = np.random.normal(128, 10, size=self.batch_size).reshape(-1, 1) * self.cell_width  # reduce sampling in x-direction
        t = np.random.uniform(self.iteration, self.tmax, size=self.batch_size).reshape(-1, 1)
        y = np.random.uniform(0, 2047, size=self.batch_size).reshape(-1, 1) * self.cell_height
        xf = np.concatenate([z, y, x, t], axis=1)
        return Tensor(xf)


