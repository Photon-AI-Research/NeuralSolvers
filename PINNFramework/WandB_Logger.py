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
        """
        wandb.log({name: scalar}, step=epoch)

    def log_image(self, image, name):
        wandb.log({name: [wandb.Image(image, caption=name)]}, step=self.epoch_counter)

    def log_plot(self, plot, name):
        wandb.log({name: plot}, step=self.epoch_counter)

    def log_histogram(self, histogram, name):
        wandb.log({"gradients": wandb.Histogram(histogram)}, step=self.epoch_counter)





