from .Logger_Interface import LoggerInterface

class PythonLogger(LoggerInterface):

    def __init__(self, logdir=None):

        self.loss_history = {}

    def log_scalar(self, scalar, name, epoch):
        """
        Add scalar data to summary.

        Args:
            scalar: the scalar to be logged
            name: name of the sclar
            epoch: epoch in the training loop
        """
        self.loss_history[name] = scalar

    def log_image(self, image, name, epoch):
        """
        Add image data to summary.
        Note that this requires the 'pillow' package.

        Args:
            image (Image) : the image tensor of shape (3,H,W) to be logged
            name (String) : name of the image
            epoch (Integer) : epoch in the training loop

        """
        self.loss_history[name] = image


    def log_plot(self, plot, name, epoch):
        """
        Logs a plot to wandb.

        Args:
            plot (plot) : the plot to be logged
            name (String) : name of the plot
            epoch (Integer) : epoch in the training loop

        """
        return

    def log_histogram(self, histogram,name, epoch):
        """
        Logs a histogram to wandb

        Args:
            histogram (histogram) : the histogram to be logged
            name (String) : name of the histogram
            epoch (Integer) : epoch in the training loop

        """
        return
