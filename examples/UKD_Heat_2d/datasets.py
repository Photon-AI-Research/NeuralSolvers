"""
Module with datasets needed to train HPM model using NeuralSolvers
(see https://github.com/ComputationalRadiationPhysics/NeuralSolvers).
Data is stored in multiple h5 files, one file for one time frame.
Each h5 file is essentially a dictionary with 2 items:
    timing (numpy array [1]): relative time point when the image was acquired [s].
    seq (numpy array [640*480]): thermal data at the time point timing.
Initializing a dataset requires argument data_info containing information about the data:
    path_data (str): path to data folder.
    num_t (int): number of time frames in the dataset.
    t_step (int): step between 2 consequent frames in the dataset.
    pix_step (int): step between 2 consequent spatieal points in the dataset.
    num_x (int): length of an image (640 for UKD data).
    num_y (int): width of an image (480 for UKD data).
    t_min (float): the time point when the first image in the dataset was acquired [s].
    t_max (float): the time point when the last image in the dataset was acquired [s].
    spat_res (float): pixel resolution of the data (approx. 0.3 for UKD data) [mm2/pixel].
"""

import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from skimage.restoration import denoise_bilateral
from skimage.filters import sobel
from skimage.segmentation import watershed
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_erosion

def load_frame(path_data, t_frame):
    """
    Upload t_frame-th image from data.
    Args:
        path_data (str): path to data folder.
        t_frame (int): number of time frame to take.
    Returns:
        value (numpy array): image data.
        timing (float): relative time point when the image was acquired [s].
    """
    # Check if given path to data is valid
    if not os.path.exists(
            path_data):
        raise FileNotFoundError('Could not find file' + path_data)
    # Upload the data
    h5_file = h5py.File(path_data + str(t_frame) + '.h5', 'r')
    value = np.array(h5_file['seq'][:])
    timing = np.array(h5_file['timing'][:])
    h5_file.close()
    return value, timing

def segmentation(path_data, t_frame, num_x, num_y, params):
    """
    Calculate segmentation mask for the brain cortex depicted at time point t_frame.
    Segmentation is calculated based on smoothing + watershed algorithm with predefined threshold.
    Args:
        path_data (str): path to data folder.
        t_frame (int): number of time frame to take.
        num_x (int): length of an image.
        num_y (int): width of an image.
        params (list): list of the from [threshold, sigma_color, sigma_spatial].
            threshold (float): absolute minimum height value used during processing.
            sigma_color (float): filter sigma in the color space. 
            sigma_spatial (float): filter sigma in the coordinate space. 
    Returns:
        seg_mask (numpy array [num_x, num_y]): binary segmentation mask.
    """
    threshold, sigma_color, sigma_spatial = params
    # Upload an image at time point t_frame
    h5_file = h5py.File(path_data + str(t_frame) + '.h5', 'r')
    value = np.array(h5_file['seq'][:])
    h5_file.close()
    # Reshape image from 1D array to 2D array
    value = np.array(value).reshape(-1)
    value = value.reshape(num_x, num_y)
    value = value.astype(np.float)
    # Apply bilateralFilter to improve segmentation quality
    value = denoise_bilateral(
        value,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        multichannel=False)
    # Segmentation algorithm
    elevation_map = sobel(value)
    markers = np.zeros_like(value)
    markers[value > threshold] = 2
    markers[value <= threshold] = 1
    seg_mask = watershed(elevation_map, markers)
    seg_mask = binary_fill_holes(seg_mask - 1)
    seg_mask = np.array(seg_mask, dtype=np.int)
    return seg_mask

class InitialConditionDataset(Dataset):
    """
    Dataset with points (x,y,t,u) to train an interpolation model u_hat: u_hat(x,y,t) ≈ u.
    """
    def __init__(self, data_info, batch_size, num_batches, segm_params):
        """Constructor of the initial condition dataset.
        Args:
            data_info (dict): dictionary with info about the data (see the module docs).
            batch_size (int): size of a mini-batch in the dataset.
            num_batches (int): number of mini-batches in the dataset.
        """
        self.u_values = []
        self.x_values = []
        self.y_values = []
        self.t_values = []
        # Check if given path to data is valid
        if not os.path.exists(
                data_info["path_data"]):
            raise FileNotFoundError(
                'Could not find file' +
                data_info["path_data"])
        # Calculate the segmentation mask for the data
        seg_mask = segmentation(
            data_info["path_data"],
            0,
            data_info["num_x"],
            data_info["num_y"],
            segm_params)
        # Load each t_step-th frame from the dataset
        for t_frame in range(
                0, data_info["num_t"], data_info["t_step"]):
            # Upload an image from dataset at time point t_frame
            u_exact, timing = load_frame(data_info["path_data"], t_frame)
            u_exact = u_exact.reshape(data_info["num_x"], data_info["num_y"]) #.astype(np.float)      
            u_exact = u_exact * seg_mask  # apply segmentation
            # Sample only each pix_step-th spatial point from an image
            for x_i in range(
                    0, data_info["num_x"], data_info["pix_step"]):
                for y_i in range(0, data_info["num_y"], data_info["pix_step"]):
                    if seg_mask[x_i, y_i] != 0:  # neglect non-cortex data
                        self.u_values.append(u_exact[x_i, y_i])
                        self.x_values.append(x_i)
                        self.y_values.append(y_i)
                        self.t_values.append(timing)
                        
        self.u_values = np.array(self.u_values).reshape(-1)
        self.x_values = np.array(self.x_values).reshape(-1)
        self.y_values = np.array(self.y_values).reshape(-1)
        self.t_values = np.array(self.t_values).reshape(-1)
        # Sometimes we are loading less files than we specified by batch_size + num_batches
        # => adapt num_batches to real number of batches for avoiding empty batches
        self.batch_size = batch_size
        num_samples = min((num_batches * batch_size, len(self.x_values)))
        self.num_batches = num_samples // self.batch_size
        # Convert grid coordinates to physical quantities ([mm])
        self.x_values = self.x_values * data_info["spat_res"]
        self.y_values = self.y_values * data_info["spat_res"]  
        # Create lists with boundary values for spatio-temporal coordinates
        self.low_bound = [
            self.x_values.min(),
            self.y_values.min(),
            self.t_values.min()]
        self.up_bound = [
            self.x_values.max(),
            self.y_values.max(),
            self.t_values.max()]
        dtype1 = torch.FloatTensor
        # Generate random permutation idx
        np.random.seed(1234)
        rand_idx = np.random.permutation(self.x_values.shape[0])
        # Permutate data points
        self.x_values = self.x_values[rand_idx]
        self.y_values = self.y_values[rand_idx]
        self.t_values = self.t_values[rand_idx]
        self.u_values = self.u_values[rand_idx]       
        # Slice data for training and convert to torch tensors
        self.x_values = dtype1(self.x_values[:num_samples])
        self.y_values = dtype1(self.y_values[:num_samples])
        self.t_values = dtype1(self.t_values[:num_samples])
        self.u_values = dtype1(self.u_values[:num_samples])
        self.low_bound = dtype1(self.low_bound)
        self.up_bound = dtype1(self.up_bound)
        
    def __len__(self):
        """
        Length of the dataset.
        """
        return self.num_batches

    def __getitem__(self, index):
        """
        Returns a mini-batch at given index containing X,u.
        Args:
            index(int): index of the mini-batch.
        Returns:
            X: spatio-temporal coordinates x,y,t concatenated.
            u: real-value function of spatio-temporal coordinates.
        """
        # Generate batch for inital solution
        x_values = (
            self.x_values[index * self.batch_size: (index + 1) * self.batch_size])
        y_values = (
            self.y_values[index * self.batch_size: (index + 1) * self.batch_size])
        t_values = (
            self.t_values[index * self.batch_size: (index + 1) * self.batch_size])
        u_values = (
            self.u_values[index * self.batch_size: (index + 1) * self.batch_size])
        return torch.stack([x_values, y_values, t_values], 1), u_values.reshape(-1,1)

class PDEDataset(Dataset):
    """
    Dataset with points (x,y,t) to train HPM model on: HPM(x,y,t) ≈ du/dt.
    """
    def __init__(self, data_info, batch_size, num_batches, segm_params):
        """Constructor of the residual poins dataset.
        Args:
            data_info (dict): dictionary with info about the data.
            batch_size (int): size of a mini-batch in the dataset.
            num_batches (int): number of mini-batches in the dataset.
        """
        self.x_values = []
        self.y_values = []
        self.t_values = []       
        seg_mask = segmentation(
            data_info["path_data"], 0, data_info["num_x"], data_info["num_y"], segm_params)
        # Consider only each t_step-th frame
        for t_frame in range(
                0, data_info["num_t"], data_info["t_step"]):
            t_frame = load_frame(data_info["path_data"], t_frame)[1]
            # Sample only each pix_step-th spatial point from the range
            for x_i in range(
                    0, data_info["num_x"], data_info["pix_step"]):
                for y_i in range(0, data_info["num_y"], data_info["pix_step"]):
                    if seg_mask[x_i, y_i] != 0:  # neglect non-cortex data
                        self.x_values.append(x_i)
                        self.y_values.append(y_i)
                        self.t_values.append(t_frame)

        self.x_values = np.array(self.x_values).reshape(-1)
        self.y_values = np.array(self.y_values).reshape(-1)
        self.t_values = np.array(self.t_values).reshape(-1)
        # Sometimes we are loading less files than we specified by batch_size + num_batches
        # => adapt num_batches to real number of batches for avoiding empty batches
        self.batch_size = batch_size
        self.num_batches = num_batches
        num_samples = min((num_batches * batch_size, len(self.x_values)))
        # Convert grid coordinates to physical quantities ([mm])
        self.x_values = self.x_values * data_info["spat_res"]
        self.y_values = self.y_values * data_info["spat_res"]
        dtype1 = torch.FloatTensor
        # Slice data for training and convert to torch tensors
        np.random.seed(1234)
        rand_idx = np.random.permutation(self.x_values.shape[0])
        # Permutate data points
        self.x_values = self.x_values[rand_idx]
        self.y_values = self.y_values[rand_idx]
        self.t_values = self.t_values[rand_idx]     
        # Slice data for training and convert to torch tensors
        self.x_values = dtype1(self.x_values[:num_samples])
        self.y_values = dtype1(self.y_values[:num_samples])
        self.t_values = dtype1(self.t_values[:num_samples])       

    def __len__(self):
        """
        Length of the dataset.
        """
        return self.num_batches

    def __getitem__(self, index):
        """
        Returns a mini-batch at given index containing X.
        Args:
            index(int): index of the mini-batch.
        Returns:
            X: spatio-temporal coordinates x,y,t concatenated.
        """
        # Generate batch with residual points
        x_values = (
            self.x_values[index * self.batch_size: (index + 1) * self.batch_size])
        y_values = (
            self.y_values[index * self.batch_size: (index + 1) * self.batch_size])
        t_values = (
            self.t_values[index * self.batch_size: (index + 1) * self.batch_size])
        return torch.stack([x_values, y_values, t_values], 1)

def derivatives(x_values, u_values):
    """
    Create an input for the HPM model.
    Args:
        x_values (torch tensor): concatenated spatio-temporal and grid coordinaties (x,y,t).
        u_values (torch tensor): real-value function to differentiate.
    Returns:
        x, y, t, u, d2u/dx2, d2u/dy2, du/dt concatenated.
    """
    # Save input in variables is necessary for gradient calculation
    x_values.requires_grad = True
    # Calculate derivatives with torch automatic differentiation
    # Move to the same device as prediction
    grads = torch.ones(u_values.shape, device=u_values.device)
    du_dx_values = torch.autograd.grad(
        u_values,
        x_values,
        create_graph=True,
        grad_outputs=grads)[0]
    #du_dx_values = [du/dx, du/dy, du/dt]
    u_x_values = du_dx_values[:, 0].reshape(u_values.shape)
    u_y_values = du_dx_values[:, 1].reshape(u_values.shape)
    u_t_values = du_dx_values[:, 2].reshape(u_values.shape)
    u_xx_values = torch.autograd.grad(
        u_x_values, x_values, create_graph=True, grad_outputs=grads)[0]
    u_yy_values = torch.autograd.grad(
        u_y_values, x_values, create_graph=True, grad_outputs=grads)[0]
    #u_xx = [u_xx, u_xy, u_xt]
    u_xx_values = u_xx_values[:, 0].reshape(u_values.shape)
    #u_yy = [u_yx, u_yy, u_yt]
    u_yy_values = u_yy_values[:, 1].reshape(u_values.shape)
    x_values, y_values, t_values = x_values.T
    
    x_values = x_values.reshape(u_values.shape)
    y_values = y_values.reshape(u_values.shape)
    t_values = t_values.reshape(u_values.shape)
    return torch.stack([x_values, y_values, t_values, u_values, u_x_values, u_y_values,
                        u_xx_values, u_yy_values, u_t_values], 1).squeeze()
