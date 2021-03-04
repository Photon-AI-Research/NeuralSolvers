import numpy as np
import openpmd_api as io
from torch import Tensor
from torch.utils.data import Dataset

class IC_Dataset(Dataset):
    
    def __init__(self, path, iteration, n0, batch_size):
        """
        Constructor of the initial condition dataset. This function loads the data with open_pmd creates the input tensos as well the labels
        
        Args:
        path: Path to the open pmd file, where the inital state is stored
        n0: defines the size of the random subset of points are used in the dataset
        iteration: defines which iteration defines the initial state
        batch_size: defines the number of points are returned by the getitem method
        """
        ## creating the iteration and save the dataset attributes
        series = io.Series(path,io.Access_Type.read_only)
        it = series.iterations[2000]
        self.n0 = n0
        self.batch_size = batch_size
        
        #loading mesh specifications
        self.cell_depth = it.get_attribute('cell_depth')
        self.cell_height = it.get_attribute('cell_height')
        self.cell_width = it.get_attribute('cell_width')
        
        #Loading fields 
        E_x = it.meshes["E"]["x"].load_chunk()
        E_y = it.meshes["E"]["y"].load_chunk()
        E_z = it.meshes["E"]["z"].load_chunk()
    
        field_shape = E_x.shape # (z, y, x)
  
        z_length = field_shape[0]
        y_length = field_shape[1]
        x_length = field_shape[2]
        
        E_x = E_x.reshape(-1,1)
        E_y = E_y.reshape(-1,1)
        E_z = E_z.reshape(-1,1)
        
        # creating the mesh in PIConGPU coordinates
        z = np.arange(0, z_length) * self.cell_depth
        y = np.arange(0, y_length) * self.cell_height
        x = np.arange(0, x_length) * self.cell_width

        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
      
        t = np.zeros(Z.shape) + (2000 * it.get_attribute("dt"))
        z = Z.reshape(-1,1)
        x = X.reshape(-1,1)
        y = Y.reshape(-1,1)
        t = t.reshape(-1,1)
        print("start concat")
        input_x = np.concatenate([z,y,x,t], axis=1)
        e_field = np.concatenate([E_z, E_y, E_x], axis=1)
        print("finished concat")
        rs = np.random.RandomState(seed=0) # create a random state for use the choice function
        rand_idx = rs.choice(input_x.shape[0], self.n0, replace=False)
        
        self.inputs = input_x[rand_idx, :]
        self.exact = e_field[rand_idx, :]
        
    
    def __len__(self):
        """
        This function returns the total number of batches which are available by the dataset
        """
        return int(self.n0 // self.batch_size)
        
        
    def  __getitem__(self, idx):
        """
        This function retuns a batch
        
        Args:
        idx: index of the batch 
        """
        x = self.inputs[self.batch_size * idx : self.batch_size * (idx + 1), :]
        exact = self.exact[self.batch_size * idx : self.batch_size * (idx + 1), :]
        
        return Tensor(x).float(), Tensor(exact).float()
        
        
        
        
        
    