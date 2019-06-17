from .callback import Callback


class MetricCallback(Callback):
    def __init__(self, metric, frequency=10):
        super().__init__(frequency=frequency, state_attribute_name=metric.name)
        self.metric = metric

    def __call__(self, trainer):
        self.metric.step(trainer.state.last_y, trainer.state.last_y_pred)
        setattr(trainer.state, self.state_attribute_name, self.metric.compute())


