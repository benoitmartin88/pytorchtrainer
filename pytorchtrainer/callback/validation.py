import torch
from .callback import Callback
from ..utils import batch_to_tensor


class ValidationCallback(Callback):
    def __init__(self, dataset_loader, metric, validate_every=10, device=None, dtype=None, non_blocking=False):
        super().__init__(frequency=validate_every, state_attribute_name="last_validation_%s" % metric.name, state_attribute_default_value=metric.default_value)
        self.dataset_loader = dataset_loader
        self.metric = metric
        self.validate_every = validate_every
        self.device = device
        self.dtype = dtype
        self.non_blocking = non_blocking

    def __call__(self, trainer):
        setattr(trainer.state, self.state_attribute_name, self._validation_function(trainer.model, trainer.prepare_batch_function))

    def _validation_function(self, model, prepare_batch_function):
        model.eval()

        device_to_use = self.device
        models_device = next(model.parameters()).device

        if self.device is None:
            # use the model's device
            device_to_use = models_device

        model.to(device_to_use)

        with torch.no_grad():
            for batch in self.dataset_loader:
                x, y, model_args = prepare_batch_function(batch, device=device_to_use, dtype=self.dtype, non_blocking=self.non_blocking)
                y_pred = model(x, **model_args)
                self.metric.step(y, y_pred)

        model.to(models_device)    # this will be a no-op if the device has not changed
        return self.metric.compute()

