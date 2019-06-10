from trainer import ModuleTrainer


class Callback(object):
    def __init__(self, state_attribute_name=None):
        self.state_attribute_name = state_attribute_name

    def __call__(self, trainer: ModuleTrainer):
        raise NotImplementedError()

    @staticmethod
    def add_state_attribute(trainer: ModuleTrainer, state_name: str, value):
        trainer.state.add_attribute(state_name, value)


from .checkpoint import SaveCheckpointCallback, LoadCheckpointCallback, SaveBestCheckpointCallback
from .file_writer import CsvWriter
from .validation import ValidationCallback

