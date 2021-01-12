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

    def __init__(self, datasets):
        super(JoinedDataset, self).__init__()
        self.datasets = datasets

    def __len__(self):
        return self.min_length(self.datasets)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
        combined_item = {}
        for key in self.datasets.keys():
            item = self.datasets[key][idx]
            combined_item[key] = item
        return combined_item

    def register_dataset(self, key, dataset):
        if key in self.datasets:
            print("Key already exists. Dataset will be overwritten")
        self.datasets[key] = dataset

