import sys
import os
import h5py
import torch
import numpy as np
from skimage.restoration import denoise_bilateral
from skimage.filters import sobel
from skimage.segmentation import watershed
from scipy.ndimage import binary_fill_holes
from argparse import ArgumentParser
from torch.utils.data import Dataset

class BoundaryConditionDataset(Dataset):

    def __init__(self, nb, lb, ub):
        """
        Constructor of the initial condition dataset
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Returns data for initial state
        """
        raise NotImplementedError

    def __len__(self):
        """
        Length of the dataset
        """
        raise NotImplementedError


class InitialConditionDataset(Dataset):

    @staticmethod
    def get2DGrid(nx, ny):
        """
        Create a vector with all postions of a 2D grid (nx X ny )
        """
        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)

        xGrid, yGrid = np.meshgrid(x, y)

        posX = xGrid.reshape(-1)
        posY = yGrid.reshape(-1)

        return posX, posY

    @staticmethod
    def get3DGrid(nx, ny, nt):
        """
        Create a vector with all postions of a 3D grid (nx X ny X nt)
        """
        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)
        t = np.arange(0, nt, 1)

        xGrid, yGrid, tGrid = np.meshgrid(x, y, t)

        posX = xGrid.reshape(-1)
        posY = yGrid.reshape(-1)
        posT = tGrid.reshape(-1)

        return posX, posY, posT

    @staticmethod
    def getInput(pData, tPoint, cSystem, spatRes=0.3):
        """
        Get the spatiotemporal coordinates for a specific time point tPoint
        The function returns a list of grid points appended with time t (nx X ny X nt)
        pData: path to a dataset
        cSystem: dictionary storing information about the dataset
        spatRes: spatial resolution of an image in the dataset [mm/pixel]
        """
        # Upload an image at time point tPoint
        # hf = {'seq': thermal data at t, 'timing': t x image length}
        hf = h5py.File(pData + str(tPoint) + '.h5', 'r')
        t = np.array(hf['timing'][0])
        hf.close()

        # Get spatial grid for an image
        posX, posY = InitialConditionDataset.get2DGrid(
            cSystem["nx"], cSystem["ny"])

        size = cSystem["nx"] * cSystem["ny"]  # number of pixels in an image
        posT = np.zeros(size) + t

        # Convert indices to physical quantities
        posX = posX * spatRes
        posY = posY * spatRes

        return posX, posY, posT

    @staticmethod
    def segmentation(pData, tPoint, nx, ny, threshold=32.4):
        """
        Calculate segmentation mask for the brain cortex depicted at time point tPoint
        pData: path to a dataset
        cSystem: dictionary storing information about the dataset
        Segmentation is calculated based on watershed algorithm with predefined threshold
        """
        # Upload an image at time point tPoint
        # hf = {'seq': thermal data at t, 'timing': t x image length}
        hf = h5py.File(pData + str(tPoint) + '.h5', 'r')
        value = np.array(hf['seq'][:])
        hf.close()

        # Reshape image from 1D array to 2D array
        value = np.array(value).reshape(-1)
        value = value.reshape(nx, ny)

        # Apply bilateralFilter to improve segmentation quality
        value = denoise_bilateral(
            value,
            sigma_color=5,
            sigma_spatial=5,
            multichannel=False)

        # Segmentation algorithm
        elevation_map = sobel(value)
        markers = np.zeros_like(value)
        markers[value > threshold] = 2
        markers[value <= threshold] = 1
        segmentation = watershed(elevation_map, markers)
        segmentation = binary_fill_holes(segmentation - 1)
        segmentation = np.array(segmentation, dtype=np.int)

        return segmentation

    @staticmethod
    def loadFrame(pData, tPoint):
        """
        Upload an image from dataset at time point tPoint
        """
        if not os.path.exists(pData):  # Check if given path to data is valid
            raise FileNotFoundError('Could not find file' + pData)

        # hf = {'seq': thermal data at t, 'timing': t x image length}
        hf = h5py.File(pData + str(tPoint) + '.h5', 'r')
        value = np.array(hf['seq'][:])
        timing = np.array(hf['timing'][:])
        hf.close()

        return value, timing

    def __init__(self, pData, batchSize, numBatches, nt, timeStep, nx=640,
                 ny=480, pixStep=4, shuffle=True, useGPU=False):
        """
        Constructor of the initial condition dataset
        __getitem()__ returns a batch with x,y,t to compute u_predicted value at as well as u_exact
        """
        self.u = []  # thermal data
        self.x = []
        self.y = []
        self.t = []

        if not os.path.exists(pData):  # Check if given path to data is valid
            raise FileNotFoundError('Could not find file' + pData)

        # Find out the last time point which data is presented at
        hf = h5py.File(pData + str(nt - 1) + '.h5', 'r')
        tmax = np.array(hf['timing'][0])
        hf.close()

        self.seg_mask = self.segmentation(
            pData=pData, tPoint=0, nx=nx, ny=ny)  # segmentation mask

        for tPoint in range(
                0, nt, timeStep):  # load each timeStep-th frame from the dataset
            # Upload an image from dataset at time point tPoint
            Exact_u, timing = self.loadFrame(pData, tPoint)
            Exact_u = Exact_u.reshape(
                nx, ny) * self.seg_mask  # apply segmentation
            for xi in range(
                    0, nx, pixStep):  # sample only each pixStep-th spatial point from an image
                for yi in range(0, ny, pixStep):
                    if Exact_u[xi, yi] != 0:  # neglect non-cortex data
                        self.u.append(Exact_u[xi, yi])
                        self.x.append(xi)
                        self.y.append(yi)
                        self.t.append(timing)

        # Convert python lists to numpy arrays
        self.u = np.array(self.u).reshape(-1)
        self.x = np.array(self.x).reshape(-1)
        self.y = np.array(self.y).reshape(-1)
        self.t = np.array(self.t).reshape(-1)

        print(len(self.x))

        # Sometimes we are loading less files than we specified by batchsize + numBatches
        # => adapt numBatches to real number of batches for avoiding empty batches
        self.batchSize = batchSize
        print("batchSize: %d" % (self.batchSize))
        self.numSamples = min((numBatches * batchSize, len(self.x)))
        print("numSamples: %d" % (self.numSamples))
        self.numBatches = self.numSamples // self.batchSize
        print("numBatches: %d" % (self.numBatches))
        self.randomState = np.random.RandomState(seed=1234)

        # Create dictionary with information about the dataset
        self.cSystem = {
            "x_lb": self.x.min(),
            "x_ub": self.x.max(),
            "y_lb": self.y.min(),
            "y_ub": self.y.max(),
            "nx": nx,
            "ny": ny,
            "nt": nt,
            "t_ub": self.t.max()}

        # Convert indices to physical quantities [mm]
        self.x = self.x * 0.25
        self.y = self.y * 0.25

        # Boundaries of spatiotemporal domain
        self.lb = np.array([self.x.min(), self.y.min(), self.t.min()])
        self.ub = np.array([self.x.max(), self.y.max(), self.t.max()])

        if (useGPU):  # send to GPU if requested
            self.dtype = torch.cuda.FloatTensor
            self.dtype2 = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype2 = torch.LongTensor

        if shuffle:  # shuffle the whole dataset if requested
            # Generate random permutation idx
            randIdx = self.randomState.permutation(self.x.shape[0])

            # Use random index
            self.x = self.x[randIdx]
            self.y = self.y[randIdx]
            self.t = self.t[randIdx]
            self.u = self.u[randIdx]

        # Slice the array for training
        self.x = self.dtype(self.x[:self.numSamples])
        self.y = self.dtype(self.y[:self.numSamples])
        self.t = self.dtype(self.t[:self.numSamples])
        self.u = self.dtype(self.u[:self.numSamples])

    def __len__(self):
        """
        Length of the dataset
        """
        return self.numBatches

    def __getitem__(self, index):
        """
        Returns item at given index
        """
        # Generate batch for inital solution
        x = (self.x[index * self.batchSize: (index + 1) * self.batchSize])
        y = (self.y[index * self.batchSize: (index + 1) * self.batchSize])
        t = (self.t[index * self.batchSize: (index + 1) * self.batchSize])
        u = (self.u[index * self.batchSize: (index + 1) * self.batchSize])
        return torch.stack([x, y, t], 1), u


class PDEDataset(Dataset):
    def __init__(self, pData, seg_mask, batchSize, numBatches, t_ub, nt, timeStep, nx=640,
                 ny=480, pixStep=4, shuffle=True, useGPU=False):
        """
        Constructor of the residual points dataset
        __getitem()__ returns a batch with x,y,t points to compute residuals at
        """
        self.x = []
        self.y = []
        self.t = []

        hf = h5py.File(pData + str(0) + '.h5', 'r')
        Exact_u = np.array(hf['seq'][:])
        hf.close()

        Exact_u = Exact_u.reshape(nx, ny) * seg_mask  # apply segmentation

        for tPoint in range(
                0, nt, timeStep):  # load each timeStep-th frame from the dataset
            # Upload an image from dataset at time point tPoint
            for xi in range(
                    0, nx, pixStep):  # sample only each pixStep-th spatial point from an image
                for yi in range(0, ny, pixStep):
                    if Exact_u[xi, yi] != 0:  # neglect non-cortex data
                        self.x.append(xi)
                        self.y.append(yi)
                        self.t.append(tPoint)

        # Convert python lists to numpy arrays
        self.x = np.array(self.x).reshape(-1)
        self.y = np.array(self.y).reshape(-1)
        self.t = np.array(self.t).reshape(-1)

        # Sometimes we are loading less files than we specified by batchsize + numBatches
        # => adapt numBatches to real number of batches for avoiding empty batches
        self.batchSize = batchSize
        print("batchSize: %d" % (self.batchSize))
        self.numSamples = min((numBatches * batchSize, len(self.x)))
        print("numSamples: %d" % (self.numSamples))
        self.numBatches = self.numSamples // self.batchSize
        print("numBatches: %d" % (self.numBatches))
        self.randomState = np.random.RandomState(seed=1234)

        # Convert indices to physical quantities [mm] & [s]
        self.x = self.x * 0.25
        self.y = self.y * 0.25
        self.t = t_ub * self.t / nt

        if (useGPU):  # send to GPU if requested
            self.dtype = torch.cuda.FloatTensor
            self.dtype2 = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype2 = torch.LongTensor

        if shuffle:  # shuffle the whole dataset if requested
            # Generate random permutation idx
            randIdx = self.randomState.permutation(self.x.shape[0])

            # Use random index
            self.x = self.x[randIdx]
            self.y = self.y[randIdx]
            self.t = self.t[randIdx]

        # Slice the array for training
        self.x = self.dtype(self.x[:self.numSamples])
        self.y = self.dtype(self.y[:self.numSamples])
        self.t = self.dtype(self.t[:self.numSamples])

    def __len__(self):
        """
        Length of the dataset
        """
        return self.numBatches

    def __getitem__(self, index):
        """
        Returns item at given index
        """
        # Generate batch with residual points
        x = (self.x[index * self.batchSize: (index + 1) * self.batchSize])
        y = (self.y[index * self.batchSize: (index + 1) * self.batchSize])
        t = (self.t[index * self.batchSize: (index + 1) * self.batchSize])
        return torch.stack([x, y, t], 1)


def derivatives(x, u):
    """
    Calculate the nn output at postion (x,y) at time t
    :param x: position
    :param t: time
    :return: Approximated solutions and their gradients
    """
    # Save input in variabeles is necessary for gradient calculation
    x.requires_grad = True

    # Calculate derivatives with torch automatic differentiation
    # Move to the same device as prediction
    grads = torch.ones(u.shape, device=u.device)
    J_U = torch.autograd.grad(u, x, create_graph=True, grad_outputs=grads)[0]
    u_x = J_U[:, 0].reshape(u.shape)
    u_y = J_U[:, 1].reshape(u.shape)
    u_t = J_U[:, 2].reshape(u.shape)

    u_xx = torch.autograd.grad(
        u_x, x, create_graph=True, grad_outputs=grads)[0]
    u_yy = torch.autograd.grad(
        u_y, x, create_graph=True, grad_outputs=grads)[0]
    u_xx = u_xx[:, 0].reshape(u.shape)
    u_yy = u_yy[:, 1].reshape(u.shape)

    x, y, t = x.T
    x = x.reshape(u.shape)
    y = y.reshape(u.shape)
    t = t.reshape(u.shape)

    return torch.stack([x, y, t, u, u_xx, u_yy, u_t], 1).squeeze()
