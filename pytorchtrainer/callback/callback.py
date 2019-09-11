
class Callback(object):
    def __init__(self, state_attribute_name=None, state_attribute_default_value=None):
        self.state_attribute_name = state_attribute_name
        self.state_attribute_default_value = state_attribute_default_value

    def __call__(self, trainer):
        raise NotImplementedError()

    def set_trainer(self, trainer):
        if self.state_attribute_name is not None:
            trainer.state.add_attribute(self.state_attribute_name, self.state_attribute_default_value)
