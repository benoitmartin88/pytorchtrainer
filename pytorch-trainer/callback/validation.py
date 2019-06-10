import torch
from callback import Callback
from utils import batch_to_tensor


class ValidationCallback(Callback):
    def __init__(self, dataset_loader, metric, validate_every=10, device=None, non_blocking=False):
        super().__init__("last_validation_%s" % metric.name)
        self.dataset_loader = dataset_loader
        self.metric = metric
        self.validate_every = validate_every
        self.device = device
        self.non_blocking = non_blocking

    def __call__(self, trainer):
        if not hasattr(trainer.state, self.state_attribute_name):
            # TODO: can this be done better ? Does it have to be done every time __call__ is called ?
            self.add_state_attribute(trainer, self.state_attribute_name, float('inf'))

        if trainer.state.current_iteration % self.validate_every == 0:
            setattr(trainer.state, self.state_attribute_name, self._validation_function(trainer.model))

    def _validation_function(self, model):
        model.eval()
        with torch.no_grad():
            for batch in self.dataset_loader:
                x, y = batch_to_tensor(batch, device=self.device, non_blocking=self.non_blocking)
                y_pred = model(x)
                self.metric.step(y, y_pred)

        return self.metric.compute()

