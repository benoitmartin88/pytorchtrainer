import torch

from . import Metric


class Accuracy(Metric):
    def __init__(self, prediction_transform=lambda x: x):
        super().__init__("accuracy", default_value=0)
        self._total_correct = 0
        self._total = 0
        self.prediction_transform = prediction_transform

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        y_pred = self.prediction_transform(y_pred)
        if y.size() != y_pred.size():
            raise TypeError("y and y_pred should have the same shape")
        correct = torch.eq(y, y_pred)

        self._total_correct += torch.sum(correct).item()
        self._total += correct.size(dim=0)    # dim 0 should be batch size

    def compute(self):
        if self._total == 0:
            raise ZeroDivisionError("Accuracy is not computable. At least one value is needed before Accuracy can be computed.")
        return self._total_correct / self._total

    def reset(self):
        self._total_correct = 0
        self._total = 0
