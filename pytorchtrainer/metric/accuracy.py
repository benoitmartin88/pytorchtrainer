import torch

from . import Metric


class Accuracy(Metric):
    def __init__(self, prediction_transform=lambda x: x):
        super().__init__("accuracy")
        self._total_correct = 0
        self._total = 0
        self.prediction_transform = prediction_transform

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        correct = torch.eq(y, self.prediction_transform(y_pred))

        self._total_correct += torch.sum(correct).item()
        self._total += correct.size(dim=0)    # dim 0 should be batch size

    def compute(self):
        if self._total == 0:
            raise ZeroDivisionError("Accuracy is not computable. At least one value is needed before Accuracy can be computed.")
        return self._total_correct / self._total

    def reset(self):
        self._total_correct = 0.
        self._total = 0
