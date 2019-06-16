import torch


class Metric(object):
    def __init__(self, name: str):
        self.name = name.replace(' ', '_')

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


from .accuracy import Accuracy
from .mean_absolute_error import MeanAbsoluteError
from .loss import Loss
