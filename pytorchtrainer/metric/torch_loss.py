import torch

from . import Metric


class TorchLoss(Metric):
    def __init__(self, loss_function: torch.nn.modules.loss):
        super().__init__("loss", default_value=float('inf'))
        self.loss_function = loss_function
        self._loss_sum = 0.

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        self._loss_sum += self.loss_function(y_pred, y).item()

    def compute(self):
        return self._loss_sum

    def reset(self):
        self._loss_sum = 0.
