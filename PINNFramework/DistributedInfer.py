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
    def __init__(self, model, save_inferences=True, dir_name="predictions", use_horovod=True, use_gpu=True):
        self.model = model
        self.save_inferences = save_inferences
        self.dir_name = dir_name
        self.use_horovod = use_horovod
        self.use_gpu = use_gpu

        if self.use_horovod:
            assert HVD_SUPPORTED, "Horovod is not supported, can not use distributed inference"

    def multi_infer(self, data):
        rank = 0
        total_processes = 1
        if self.use_horovod:
            # Initialize Horovod
            hvd.init()
            # Pin GPU to be used to process local rank (one GPU per process)
            torch.cuda.set_device(hvd.local_rank())
            rank = hvd.rank()
            total_processes = hvd.size()

        # split the data and assign the data according to the horovod ranks
        datasize_per_process = math.ceil(data.shape[0] / total_processes)

        if self.use_gpu:
            infer_data = data[(datasize_per_process*rank): datasize_per_process].float().cuda()
        else:
            infer_data = torch.split(data, datasize_per_process)[rank]

        # get the predictions
        if not rank:
            print("Full data size", data.shape)
            print("Rank-0 data size:", infer_data.shape)
            print("Data size per process:", datasize_per_process)
        sub_prediction = self.model(infer_data)

        # save all the inferences in separate file
        if self.save_inferences:
            # create the dictionary to save the predictions
            if not rank:
                if not os.path.exists(self.dir_name):
                    os.mkdir(self.dir_name)
            else:
                while not os.path.exists(self.dir_name):
                    pass

            # save the predictions
            sub_prediction = sub_prediction.detach().cpu().numpy()
            np.save(os.path.join(self.dir_name, "pred_"+str(rank)), sub_prediction)

            # process-0 informs that files saved successfully
            # FUTURE: get data from the all node to node-0
            if not rank:
                print("all prediction files has been saved in the directory: ", self.dir_name)
        else:
            return sub_prediction
