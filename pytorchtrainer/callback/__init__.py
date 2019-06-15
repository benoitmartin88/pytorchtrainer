from .. import ModuleTrainer


class Callback(object):
    def __init__(self, frequency=1, state_attribute_name=None):
        if frequency < 0:
            raise ValueError("frequency argument should be positive.")
        self.frequency = frequency
        self.state_attribute_name = state_attribute_name

    def __call__(self, trainer: ModuleTrainer):
        raise NotImplementedError()

    @staticmethod
    def add_state_attribute(trainer: ModuleTrainer, state_name: str, value):
        trainer.state.add_attribute(state_name, value)


from .checkpoint import SaveCheckpointCallback, LoadCheckpointCallback, SaveBestCheckpointCallback
from .file_writer import CsvWriter
from .validation import ValidationCallback

