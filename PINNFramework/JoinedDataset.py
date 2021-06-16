from torch.utils.data import Dataset


class JoinedDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.
        datasets (sequence): List of datasets to be concatenated
    """
    @staticmethod
    def min_length(datasets):
        """
        Calculates the minimum dataset length of a list of datasets

        datasets (Map): Map of datasets to be concatenated
        """
        minimum = float("inf")
        for key in datasets.keys():
            length = len(datasets[key])
            if length < minimum:
                minimum = length
        return minimum

    @staticmethod
    def max_length(datasets):
        """
        Calculates the minimum dataset length of a list of datasets

        datasets (Map): Map of datasets to be concatenated
        """
        maximum = -1 * float("inf")
        for key in datasets.keys():
            length = len(datasets[key])
            if length > maximum:
                maximum = length
        return maximum

    def __init__(self, datasets, mode='min'):
        super(JoinedDataset, self).__init__()
        self.datasets = datasets
        self.mode = mode

    def __len__(self):
        if self.mode =='min':
            return self.min_length(self.datasets)
        if self.mode =='max':
            return self.max_length(self.datasets)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
        combined_item = {}
        for key in self.datasets.keys():
            if self.mode == 'max':
                idx = idx % len(self.datasets[key])
            item = self.datasets[key][idx]
            combined_item[key] = item
        return combined_item

    def register_dataset(self, key, dataset):
        if key in self.datasets:
            print("Key already exists. Dataset will be overwritten")
        self.datasets[key] = dataset

