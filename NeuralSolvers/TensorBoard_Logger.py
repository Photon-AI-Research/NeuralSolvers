from .Logger_Interface import LoggerInterface
from tensorboardX import SummaryWriter

class TensorBoardLogger(LoggerInterface):

    def __init__(self, logdir=None):
        """
        Create an event file in a given directory.

        Args:
            logdir: save directory location
        """
        self.writer = SummaryWriter(logdir)

    def log_scalar(self, scalar, name, epoch):
        """
        Add scalar data to summary.

        Args:
            scalar: the scalar to be logged
            name: name of the sclar
            epoch: epoch in the training loop
        """
        self.writer.add_scalar(name, scalar, epoch)

    def log_image(self, image, name, epoch):
        """
        Add image data to summary. 
        Note that this requires the 'pillow' package.

        Args:
            image (Image) : the image tensor of shape (3,H,W) to be logged
            name (String) : name of the image
            epoch (Integer) : epoch in the training loop

        """
        self.writer.add_image(name, image, epoch)


    def log_plot(self, plot, name, epoch):
        """
        Logs a plot to wandb.

        Args:
            plot (plot) : the plot to be logged
            name (String) : name of the plot
            epoch (Integer) : epoch in the training loop

        """
        self.writer.add_figure(name, plot, epoch)

    def log_histogram(self, histogram,name, epoch):
        """
        Logs a histogram to wandb

        Args:
            histogram (histogram) : the histogram to be logged
            name (String) : name of the histogram
            epoch (Integer) : epoch in the training loop

        """
        self.writer.add_histogram(name, histogram, epoch)
