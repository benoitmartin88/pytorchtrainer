import torch
import torch.nn as nn
import torch.optim as optim
import os

from .callback import Callback

default_save_directory = './checkpoint'
default_filename = 'checkpoint.pt.tar'
default_best_filename = 'best.pt.tar'


class SaveCheckpointCallback(Callback):
    def __init__(self, save_directory=default_save_directory, filename=default_filename):
        super().__init__()
        self.save_directory = save_directory
        self.filename = filename

        os.makedirs(self.save_directory, exist_ok=True)

    def __call__(self, trainer):
        self._save_checkpoint(trainer.model, trainer.optimizer, trainer.state)

    def _save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, trainer_state):
        # from .. import __version__

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'trainer_state': trainer_state,
            'check_model_class': str(model.__class__),
            'check_optimizer_class': str(optimizer.__class__),
            # 'check_trainer_version': __version__    # TODO
        }, os.path.join(self.save_directory, self.filename))


class LoadCheckpointCallback(Callback):
    def __init__(self, save_directory=default_save_directory, filename=default_filename, callback=None):
        super().__init__()
        self.save_directory = save_directory
        self.filename = filename
        self.callback = callback

    def __call__(self, trainer):
        self._load_checkpoint(trainer.model, trainer.optimizer, trainer.state)
        if self.callback is not None:
            if not callable(self.callback):
                raise TypeError("Argument callback should be a function.")
            self.callback(trainer)

    def _load_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, state):
        # from .. import __version__

        checkpoint = torch.load(os.path.join(self.save_directory, self.filename))

        # checks
        if checkpoint['check_model_class'] != str(model.__class__):
            raise TypeError("Models do not match: %s and %s" % (checkpoint['check_model_class'], model.__class__))
        if checkpoint['check_optimizer_class'] != str(optimizer.__class__):
            raise TypeError("Optimizers do not match: %s and %s" % (checkpoint['check_optimizer_class'], optimizer.__class__))

        # checkpoint['check_trainer_version']   # TODO

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        state.set(checkpoint['trainer_state'])


class SaveBestCheckpointCallback(SaveCheckpointCallback):
    def __init__(self, state_metric_name: str, saves_to_keep=5, comparison_function=lambda metric, best: metric < best,
                 save_directory=default_save_directory, filename=default_best_filename,
                 filename_transform_function=None):
        super().__init__(save_directory, filename)
        self.state_metric_name = state_metric_name
        # self.saves_to_keep = saves_to_keep    # TODO
        self.comparison_function = comparison_function
        self.current_best = None

        if filename_transform_function is None:
            self.filename_transform_function = self._default_filename_transform_function

        os.makedirs(self.save_directory, exist_ok=True)

    def __call__(self, trainer):
        """
        best.pt.tar -> best_EPOCH_METRIC_1.pt.tar
        :param trainer:
        :return:
        """
        if self.current_best is None or self.comparison_function(trainer.state.last_train_loss, self.current_best):
            self.current_best = trainer.state.get(self.state_metric_name)

            old_filename = self.filename
            self.filename = self.filename_transform_function(self.filename, trainer.state)

            self._save_checkpoint(trainer.model, trainer.optimizer, trainer.state)
            self.filename = old_filename

    @staticmethod
    def _default_filename_transform_function(filename, state):
        c = filename.count('.')
        base, *ext = filename.rsplit('.', c)
        return base + "_%d_%.2f_%d." % (state.current_epoch, state.last_train_loss, 1) + '.'.join(ext)

