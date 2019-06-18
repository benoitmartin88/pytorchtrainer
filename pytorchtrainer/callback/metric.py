from .callback import Callback


class MetricCallback(Callback):
    def __init__(self, metric, frequency=1):
        super().__init__(frequency=frequency, state_attribute_name=metric.name, state_attribute_default_value=metric.default_value)
        self.metric = metric
        self.last_epoch = 0

    def __call__(self, trainer):
        if self.last_epoch != trainer.state.current_epoch:
            self.metric.reset()

        self.last_epoch = trainer.state.current_epoch

        self.metric.step(trainer.state.last_y, trainer.state.last_y_pred)
        setattr(trainer.state, self.state_attribute_name, self.metric.compute())


