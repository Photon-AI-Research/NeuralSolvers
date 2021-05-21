from abc import ABC, abstractmethod


class LoggerInterface(ABC):

    @abstractmethod
    def log_scalar(self, scalar, name, epoch):
        """
        Method that defines how scalars are logged

        Args:
            scalar: scalar to be logged

        """
        pass

    @abstractmethod
    def log_image(self, image, name):
        """
        Method that defines how images are logged

        Args:
            image: image to be logged

        """
        pass

    @abstractmethod
    def log_histogram(self, histogram, name):
        """
        Method that defines how images are logged

        Args:
            histogram: histogram to be logged

        """
        pass


