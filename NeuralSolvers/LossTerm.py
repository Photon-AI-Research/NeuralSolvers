from torch.nn import L1Loss, MSELoss


class LossTerm:
    """
    Defines the main structure of a loss term
    """
    def __init__(self, dataset, name, norm='L2', weight=1.):
        """
        Constructor of a loss term
        
        Args:
            dataset (torch.utils.Dataset): dataset that provides the residual points
            norm: Norm used for calculation PDE loss
            weight: Weighting for the loss term
        """
        # cases for standard torch norms
        if norm == 'L2':
            self.norm = MSELoss()
        elif norm == 'L1':
            self.norm = L1Loss()
        else:
            # Case for self implemented norms
            self.norm = norm
        self.dataset = dataset
        self.name = name
        self.weight = weight

