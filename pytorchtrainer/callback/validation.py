import torch
from .callback import Callback
from ..utils import batch_to_tensor


class ValidationCallback(Callback):
    def __init__(self, dataset_loader, metric, validate_every=10, device=None, dtype=None, non_blocking=False):
        super().__init__(frequency=validate_every, state_attribute_name="last_validation_%s" % metric.name)
        self.dataset_loader = dataset_loader
        self.metric = metric
        self.validate_every = validate_every
        self.device = device
        self.dtype = dtype
        self.non_blocking = non_blocking

    def __call__(self, trainer):
        setattr(trainer.state, self.state_attribute_name, self._validation_function(trainer.model))

    def _validation_function(self, model):
        model.eval()
        with torch.no_grad():
            for batch in self.dataset_loader:
                x, y = batch_to_tensor(batch, device=self.device, dtype=self.dtype, non_blocking=self.non_blocking)
                y_pred = model(x)
                self.metric.step(y, y_pred)

        return self.metric.compute()
