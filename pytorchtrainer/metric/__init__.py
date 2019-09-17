import torch


class Metric(object):
    def __init__(self, name: str, default_value=None):
        self.name = name.replace(' ', '_')
        self.default_value = default_value

    def step(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


from .accuracy import Accuracy
from .mean_absolute_error import MeanAbsoluteError
from .torch_loss import TorchLoss
