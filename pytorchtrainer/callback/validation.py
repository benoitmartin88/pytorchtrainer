from .callback import Callback


class ValidationCallback(Callback):
    def __init__(self, dataloader, metric, device=None, dtype=None, non_blocking=False):
        super().__init__(state_attribute_name="last_validation_%s" % metric.name, state_attribute_default_value=metric.default_value)
        self.dataloader = dataloader
        self.metric = metric
        self.device = device
        self.dtype = dtype
        self.non_blocking = non_blocking

    def __call__(self, trainer):
        setattr(trainer.state, self.state_attribute_name, trainer.evaluate(self.dataloader, self.metric))
