from torch.nn import Module
from NeuralSolvers.loggers.Logger_Interface import LoggerInterface


class Callback:
    def __init__(self):
        self.model = None
        self.logger = None

    def set_model(self, model):
        if isinstance(model, Module):
            self.model = model
        else:
            raise ValueError("Model is not of type <torch.nn.module> but model of type {} was found"
                             .format(type(model)))

    def set_logger(self, logger):
        if isinstance(LoggerInterface):
            self.logger
        else:
            raise ValueError("Logger is not of type <LoggerInterface> but logger of type {} was found"
                             .format(type(logger)))

    def __call__(self, epoch):
        raise NotImplementedError("method __call__() of the callback is not implemented")


class CallbackList:
    def __init__(self, callbacks):
        if isinstance(callbacks, list):
            for cb in callbacks:
                if not isinstance(cb, Callback):
                    raise ValueError("Callback has to be of type <Callback> but type {} was found"
                                     .format(type(cb)))
            self.callbacks = callbacks
        else:
            raise ValueError("Callback has to be of type <list> but type {} was found"
                             .format(type(callbacks)))

    def __call__(self, epoch):
        for cb in self.callbacks:
            cb(epoch)






