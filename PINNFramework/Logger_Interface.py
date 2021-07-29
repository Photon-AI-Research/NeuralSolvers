from abc import ABC, abstractmethod


class LoggerInterface(ABC):

    @abstractmethod
    def log_scalar(self, scalar, name, epoch):
        """
        Method that defines how scalars are logged

        Args:
            scalar: scalar to be logged
            name: name of the scalar
            epoch: epoch in the training loop

        """
        pass

    @abstractmethod
    def log_image(self, image, name, epoch):
        """
        Method that defines how images are logged

        Args:
            image: image to be logged
            name: name of the image
            epoch: epoch in the training loop

        """
        pass

    @abstractmethod
    def log_histogram(self, histogram, name, epoch):
        """
        Method that defines how images are logged

        Args:
            histogram: histogram to be logged
            name: name of the histogram
            epoch: epoch in the training loop

        """
        pass


