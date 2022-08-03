import torch
import math
import os
import numpy as np

try:
    import horovod.torch as hvd

    print("Local rank / Global Rank : ", hvd.local_rank(), "/", hvd.rank())
    HVD_SUPPORTED = True
except:
    print("Was not able to import Horovod. Thus Horovod support is not enabled")
    HVD_SUPPORTED = False


class DistributedInfer:
    def __init__(self, model, save_inferences=True, dir_name="predictions", use_horovod=True):
        """
            Constructor of the Distributed-Inference class

            Args:
                model (torch.nn.Module): the model to be used for inference
                save_inferences (bool): if true, the predictions will be saved in a directory
                dir_name (str): name of the directory to save the predictions
                use_horovod (bool): if true, the inference will be done using Horovod
                use_gpu (bool): if true, the inference will be done using GPU
        """
        self.model = model
        self.save_inferences = save_inferences
        self.dir_name = dir_name
        self.use_horovod = use_horovod

        self.rank = 0
        self.total_processes = 1
        if self.use_horovod:
            assert HVD_SUPPORTED, "Horovod is not supported, can not use distributed inference"
            if self.use_horovod:
                # Initialize Horovod
                hvd.init()
                # Pin GPU to be used to process local rank (one GPU per process)
                torch.cuda.set_device(hvd.local_rank())
                self.rank = hvd.rank()
                self.total_processes = hvd.size()

    def multi_infer(self, data):
        """
            This function will perform the inference on the data using multiple GPUs
            Args:
                data (torch.Tensor): the total data to be used for inference
        """
        # split the data and assign the data according to the horovod ranks
        datasize_per_process = math.ceil(data.shape[0] / self.total_processes)

        end_idx = datasize_per_process * (self.rank + 1)
        if end_idx > data.shape[0]:
            end_idx = data.shape[0]
        infer_data = data[(datasize_per_process * self.rank): end_idx].float().cuda()

        # get the predictions
        if not self.rank:
            print("Full data size", data.shape)
            print("Rank-0 data size:", infer_data.shape)
            print("Data size per process:", datasize_per_process)
        sub_prediction = self.model(infer_data)

        # save all the inferences in separate file
        if self.save_inferences:
            # create the dictionary to save the predictions
            if not self.rank:
                if not os.path.exists(self.dir_name):
                    os.mkdir(self.dir_name)
            else:
                while not os.path.exists(self.dir_name):
                    pass

            # save the predictions
            sub_prediction = sub_prediction.detach().cpu().numpy()
            np.save(os.path.join(self.dir_name, "pred_" + str(self.rank)), sub_prediction)

            # process-0 informs that files saved successfully
            # FUTURE: get data from the all node to node-0
            if not self.rank:
                print("all prediction files has been saved in the directory: ", self.dir_name)
        return sub_prediction

    def combine_numpy_files(self, num_of_files=0):
        """
            This function will combine all the numpy files in the directory into one numpy file
            Args:
                num_of_files (int)(optional): number of files to be combined
        """
        # combine the numpy files
        if num_of_files == 0:
            num_of_files = self.total_processes
        if not self.rank:
            print("Combining the output files")
            numpy_files = [os.path.join(self.dir_name, "pred_" + str(i) + ".npy") for i in range(num_of_files)]
            combined_file_name = os.path.join(self.dir_name, "predictions.npy")
            combined_data = np.concatenate([np.load(f) for f in numpy_files])
            np.save(combined_file_name, combined_data)
            print("Combined file has been saved at :", combined_file_name)
            return combined_data
