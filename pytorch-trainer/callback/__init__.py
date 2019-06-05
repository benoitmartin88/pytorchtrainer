from trainer import ModuleTrainer


class Callback(object):
    def __call__(self, trainer: ModuleTrainer):
        raise NotImplementedError()


from .checkpoint import SaveCheckpointCallback, LoadCheckpointCallback
from .file_writer import CsvWriter
from .validation import ValidationCallback

