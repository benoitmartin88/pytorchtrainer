import os
import csv
import time

from .callback import Callback

default_save_directory = './log'
default_filename = 'log.cvs'


class CsvWriter(Callback):
    def __init__(self, save_directory=default_save_directory, filename=default_filename, delimiter=';',
                 extra_header=None, callback=None):
        super().__init__()
        file, ext = filename.rsplit('.', 1)
        self.log_file_path = os.path.join(save_directory, file + '_' + time.strftime("%Y%m%d_%H%M%S") + '.' + ext)
        self.delimiter = delimiter
        self.callback = callback

        os.makedirs(save_directory, exist_ok=True)

        with open(self.log_file_path, mode='w') as writer:
            header = ['timestamp', 'epoch', 'iteration', 'train loss']
            if extra_header is not None:
                if not isinstance(extra_header, list):
                    raise TypeError("extra_header should be a list.")
                header += extra_header

            writer = csv.writer(writer, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)

    def __call__(self, trainer):
        self.__save(trainer.state)

    def __save(self, trainer_state):
        with open(self.log_file_path, mode='a') as writer:
            writer = csv.writer(writer, delimiter=self.delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)

            extra = []
            if self.callback is not None:
                extra = self.callback(trainer_state)
                if not isinstance(extra, list):
                    raise TypeError("callback should return a list.")

            writer.writerow([time.time(), trainer_state.current_epoch+1, trainer_state.current_iteration+1, trainer_state.last_train_loss] + extra)
