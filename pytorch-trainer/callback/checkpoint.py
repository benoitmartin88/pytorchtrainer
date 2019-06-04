import torch
import os

from callback import Callback


default_save_diretory = './checkpoint'
default_filename = 'checkpoint.pt.tar'


class SaveCheckpointCallback(Callback):
    def __init__(self, save_every=10, save_directory=default_save_diretory, filename=default_filename):
        self.save_every = save_every
        self.save_directory = save_directory
        self.filename = filename

    def __call__(self, trainer):
        if trainer.state.current_epoch % self.save_every == 0:
            self._save_checkpoint(trainer.model, trainer.optimizer, trainer.state)

    def _save_checkpoint(self, model, optimizer, trainer_state):
        # from .. import __version__

        os.makedirs(self.save_directory, exist_ok=True)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'trainer_state': trainer_state,
            # 'trainer_version': __version__    # TODO
        }, os.path.join(self.save_directory, self.filename))


class LoadCheckpointCallback(Callback):
    def __init__(self, save_directory=default_save_diretory, filename=default_filename):
        self.save_directory = save_directory
        self.filename = filename

    def __call__(self, trainer):
        self._load_checkpoint(trainer.model, trainer.optimizer, trainer.state)

    def _load_checkpoint(self, model, optimizer, checkpoint_dir='./checkpoint', filename='checkpoint.pt.tar'):
        # from .. import __version__

        checkpoint = torch.load(os.path.join(checkpoint_dir, filename))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.state = checkpoint['trainer_state']
        # saved_trainer_version = checkpoint['trainer_version']   # TODO
