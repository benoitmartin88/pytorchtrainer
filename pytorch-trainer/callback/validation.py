import torch
from callback import Callback
from utils import batch_to_tensor


class ValidationCallback(Callback):
    def __init__(self, dataset_loader, metric, validate_every=10, device=None, non_blocking=False):
        self.dataset_loader = dataset_loader
        self.metric = metric
        self.validate_every = validate_every
        self.device = device
        self.non_blocking = non_blocking

    def __call__(self, trainer):
        if trainer.state.current_iteration % self.validate_every == 0:
            trainer.state.last_validation_loss = self._validation_function(trainer.model)

    def _validation_function(self, model):
        model.eval()
        with torch.no_grad():
            for batch in self.dataset_loader:
                x, y = batch_to_tensor(batch, device=self.device, non_blocking=self.non_blocking)
                y_pred = model(x)
                self.metric.step(y, y_pred)

        return self.metric.compute()

