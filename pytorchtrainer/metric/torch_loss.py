import torch

from . import Metric


class TorchLoss(Metric):
    def __init__(self, loss_function: torch.nn.modules.loss):
        super().__init__("loss", default_value=float('inf'))
        self.loss_function = loss_function
        self._loss_sum = 0.
        self._total = 0

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        loss = self.loss_function(y_pred, y)
        self._loss_sum += loss.item()
        self._total += 1
        return loss

    def compute(self):
        if self._total == 0:
            raise ZeroDivisionError("Loss average is not computable.")
        return self._loss_sum / self._total

    def reset(self):
        self._loss_sum = 0.
        self._total = 0
