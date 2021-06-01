from .Logger_Interface import LoggerInterface
import wandb


class WandbLogger(LoggerInterface):

    def __init__(self, project, args, entity=None):
        """
        Initialize wandb instance and connect to the server

        Args:
            project: name of the project
            args: hyperparameters used for this runs
            writing_cycle: defines the writing period
            entity: account or group id used for that run
        """
        wandb.init(project=project, entity=entity)
        wandb.config.update(args)  # adds all of the arguments as config variable

    def log_scalar(self, scalar, name, epoch):
        """
        Logs a scalar to wandb

        Args:
            scalar: the scalar to be logged
            name: name of the sclar
            epoch: epoch in the training loop
        """
        wandb.log({name: scalar}, step=epoch)

    def log_image(self, image, name, epoch):
        """
        Logs a image to wandb

        Args:
            image (Image) : the image to be logged
            name (String) : name of the image
            epoch (Integer) : epoch in the training loop

        """
        wandb.log({name: [wandb.Image(image, caption=name)]}, step=epoch)

    def log_plot(self, plot, name, epoch):
        """
        Logs a plot to wandb

        Args:
            plot (plot) : the plot to be logged
            name (String) : name of the plot
            epoch (Integer) : epoch in the training loop

        """
        wandb.log({name: plot}, step=epoch)

    def log_histogram(self, histogram,name, epoch):
        """
        Logs a histogram to wandb

        Args:
            histogram (histogram) : the histogram to be logged
            name (String) : name of the histogram
            epoch (Integer) : epoch in the training loop

        """
        wandb.log({name: wandb.Histogram(histogram)}, step=epoch)





