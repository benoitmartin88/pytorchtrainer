import torch.nn as nn
import torch.optim as optim
import os
import csv
import time

from callback import Callback

default_save_diretory = './log'
default_filename = 'log.cvs'


class CsvWriter(Callback):
    def __init__(self, save_every=10, save_directory=default_save_diretory, filename=default_filename, delimiter=';',
                 extra_header=None, callback=None):
        self.save_every = save_every
        file, ext = filename.rsplit('.', 1)
        self.log_file_path = os.path.join(save_directory, file + '_' + time.strftime("%Y%M%d_%H%M%S") + '.' + ext)
        self.delimiter = delimiter
        self.callback = callback

        os.makedirs(save_directory, exist_ok=True)

        with open(self.log_file_path, mode='w') as writer:
            header = ['timestamp', 'epoch', 'train loss', 'validation loss']
            if extra_header is not None:
                if not isinstance(extra_header, list):
                    raise TypeError("extra_header should be a list.")
                header += extra_header

            writer = csv.writer(writer, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)

    def __call__(self, trainer):
        if trainer.state.current_epoch % self.save_every == 0:
            self.__save(trainer.model, trainer.optimizer, trainer.state)

    def __save(self, model: nn.Module, optimizer: optim.Optimizer, trainer_state):
        with open(self.log_file_path, mode='a') as writer:
            writer = csv.writer(writer, delimiter=self.delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)

            extra = []
            if self.callback is not None:
                extra = self.callback(self)
                if not isinstance(extra, list):
                    raise TypeError("callback should return a list.")

            writer.writerow([time.time(), trainer_state.current_epoch, trainer_state.last_train_loss, trainer_state.last_validation_loss] + extra)
