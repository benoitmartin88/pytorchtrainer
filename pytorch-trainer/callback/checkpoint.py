import torch
import torch.nn as nn
import torch.optim as optim
import os

from callback import Callback
from trainer import State

default_save_diretory = './checkpoint'
default_filename = 'checkpoint.pt.tar'


class SaveCheckpointCallback(Callback):
    def __init__(self, save_every=10, save_directory=default_save_diretory, filename=default_filename):
        self.save_every = save_every
        self.save_directory = save_directory
        self.filename = filename

        os.makedirs(self.save_directory, exist_ok=True)

    def __call__(self, trainer):
        if trainer.state.current_epoch % self.save_every == 0:
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
    def __init__(self, save_directory=default_save_diretory, filename=default_filename, callback=None):
        self.save_directory = save_directory
        self.filename = filename
        self.callback = callback

    def __call__(self, trainer):
        self._load_checkpoint(trainer.model, trainer.optimizer, trainer.state)
        if self.callback is not None:
            if not callable(self.callback):
                raise TypeError("Argument callback should be a function.")
            self.callback(trainer)

    def _load_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, state: State):
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
