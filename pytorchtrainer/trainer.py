import collections
import datetime
from time import time
import signal

import torch.nn as nn
import torch.optim as optim

from .callback import Callback
from .stop_condition import NoStopping
from .utils import print_progress, batch_to_tensor


class State(object):
    current_epoch = 0
    current_iteration = 0
    last_x = None
    last_y = None
    last_y_pred = None
    last_train_loss = float("inf")

    def set(self, state):
        if not isinstance(state, State):
            raise TypeError("state argument should be of type State.")

        for k, v in state.__dict__.items():
            setattr(self, k, v)

    def add_attribute(self, attribute_name: str, value):
        if not hasattr(self, attribute_name):
            setattr(self, attribute_name, value)

    def get(self, attribute_name: str):
        return getattr(self, attribute_name)


class ModuleTrainer(object):
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, train_function,
                 stop_condition=NoStopping(),
                 init_callback=None):
        if not callable(train_function):
            raise TypeError("Argument compute_function should be a function.")

        if not callable(stop_condition):
            raise TypeError("Argument stop_condition should be a function.")

        signal.signal(signal.SIGINT, self.__graceful_exit)

        self.state = State()
        self.model = model
        self.optimizer = optimizer
        self.train_function = train_function
        self.stop_condition = stop_condition

        self.__post_iteration_callback = []
        self.__post_epoch_callback = []

        self.extra_progressbar_metrics = ()

        if init_callback is not None:
            if not callable(init_callback):
                raise TypeError("Argument post_init_callback should be a function.")

            init_callback(self)

    def train(self, train_dataset_loader, max_epochs=100, verbose=1):
        self.model.train()  # set the module to training mode

        train_start = time()
        while self.state.current_epoch < max_epochs and not self.stop_condition(self.state):
            self.model.zero_grad()

            for self.state.current_iteration, batch in enumerate(train_dataset_loader):
                iteration_start = time()

                # Run the actual compute function
                self.state.last_x, self.state.last_y, self.state.last_y_pred, self.state.last_train_loss = self.train_function(batch)

                self.__run_post_iteration_callbacks()

                iteration_elapsed_time = time() - iteration_start

                if verbose == 1:
                    self.__update_progress_bar(iteration_elapsed_time, len(train_dataset_loader), max_epochs)

            self.state.current_epoch += 1
            self.__run_post_epoch_callbacks()

        print("train time %.2f" % (time() - train_start))

    def add_progressbar_metric(self, format: str, metric_state_names: list):
        if not isinstance(format, str) or not isinstance(metric_state_names, list):
            raise TypeError("format should be string (eg: '%.2f') and metric_state_names should be a list of metric names (eg: 'accuracy')")
        self.extra_progressbar_metrics = (format, metric_state_names)

    def register_post_iteration_callback(self, callback: Callback):
        callback.set_trainer(self)  # This is done here to simplify the API by hiding this from the user
        self.__post_iteration_callback.append(callback)

    def register_post_epoch_callback(self, callback: Callback):
        callback.set_trainer(self)    # This is done here to simplify the API by hiding this from the user
        self.__post_epoch_callback.append(callback)

    def __graceful_exit(self, signum, frame):
        print("Sig %s caught. Graceful exit has been called. Currently running epoch will be finished." % signum)
        self.stop_condition = lambda state: True

    def __run_post_iteration_callbacks(self):
        for cb in self.__post_iteration_callback:
            if self.state.current_iteration % cb.frequency == 0:
                cb(self)

    def __run_post_epoch_callbacks(self):
        for cb in self.__post_epoch_callback:
            if cb.frequency != 0 and self.state.current_epoch % cb.frequency == 0:
                cb(self)

    def __update_progress_bar(self, iteration_elapsed_time, train_dataset_loader_size, max_epochs):
        remaining_time_estimation = int(iteration_elapsed_time * (train_dataset_loader_size - self.state.current_iteration))

        if len(self.extra_progressbar_metrics) > 0:
            suffix = ("%d/%d | %.2f s/it | %s remaining | train loss %.4f | " + self.extra_progressbar_metrics[0]) % \
                     (self.state.current_iteration + 1, train_dataset_loader_size, iteration_elapsed_time,
                      str(datetime.timedelta(seconds=remaining_time_estimation)), self.state.last_train_loss,
                      *[getattr(self.state, a) for a in self.extra_progressbar_metrics[1]])
        else:
            suffix = "%d/%d | %.2f s/it | %s remaining | train loss %.4f" % \
                     (self.state.current_iteration + 1, train_dataset_loader_size, iteration_elapsed_time,
                      str(datetime.timedelta(seconds=remaining_time_estimation)), self.state.last_train_loss)

        print_progress(self.state.current_iteration + 1, train_dataset_loader_size,
                       bar_length=25,
                       prefix="epoch %d/%d" % (self.state.current_epoch + 1, max_epochs),
                       suffix=suffix)


def create_default_trainer(model: nn.Module, optimizer: optim.Optimizer, criterion,
                           device=None, dtype=None, non_blocking=False,
                           stop_condition=NoStopping(),
                           prepare_batch=batch_to_tensor,
                           output_transform=lambda x, y, y_pred, loss: (x, y, y_pred, loss.item()),
                           init_callback=None):

    if not callable(prepare_batch):
        raise TypeError("Argument prepare_batch should be a function.")
    if not callable(output_transform):
        raise TypeError("Argument output_transform should be a function.")

    if device:
        model.to(device)

    if dtype:
        model.to(dtype=dtype)

    # reset optimizer state
    optimizer.state = collections.defaultdict(dict)

    def _default_train_function(batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, dtype=dtype, non_blocking=non_blocking)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return ModuleTrainer(model, optimizer,
                         train_function=_default_train_function,
                         stop_condition=stop_condition,
                         init_callback=init_callback)