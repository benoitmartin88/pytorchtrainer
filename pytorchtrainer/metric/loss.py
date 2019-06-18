import torch

from . import Metric


class Loss(Metric):
    def __init__(self, loss_function):
        super().__init__("loss", default_value=float('inf'))
        self.loss_function = loss_function
        self._loss_sum = 0.
        self._total = 0

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        self._loss_sum += self.loss_function(y_pred, y).item()
        self._total += y.size(dim=0)  # dim 0 should be batch size

    def compute(self):
        if self._total == 0:
            raise ZeroDivisionError("Loss average is not computable.")
        return self._loss_sum / self._total

    def reset(self):
        self._loss_sum = 0.
        self._total = 0
