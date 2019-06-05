import torch

from metric import Metric


class MeanAbsoluteError(Metric):
    def __init__(self):
        self._absolute_error_sum = 0.
        self._total = 0

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        absolute_errors = torch.abs(y - y_pred)
        self._absolute_error_sum += torch.sum(absolute_errors).item()
        self._total += y.size(dim=0)    # dim 0 should be batch size

    def compute(self):
        if self._total == 0:
            raise ZeroDivisionError("Mean absolute error is not computable.")
        return self._absolute_error_sum / self._total

    def reset(self):
        self._absolute_error_sum = 0.
        self._total = 0
