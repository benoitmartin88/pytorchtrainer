import datetime
from time import time

import torch
import torch.nn as nn
import torch.optim as optim

from stop_condition import NoStopping
from utils import print_progress


class State(object):
    current_epoch = 0
    last_loss = float("inf")


class ModuleTrainer(object):
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, compute_function,
                 stop_condition=NoStopping(),
                 init_callback=None):
        if not callable(compute_function):
            raise TypeError("Argument compute_function should be a function.")

        if not callable(stop_condition):
            raise TypeError("Argument stop_condition should be a function.")

        self.model = model
        self.optimizer = optimizer
        self.compute_function = compute_function
        self.state = State
        self.stop_condition = stop_condition

        self.post_epoch_callback = []

        if init_callback is not None:
            if not callable(init_callback):
                raise TypeError("Argument post_init_callback should be a function.")

            init_callback(self)

    def train(self, train_dataset_loader, max_epochs=100, verbose=1):
        self.model.train()  # set the module to training mode

        train_start = time()
        while self.state.current_epoch < max_epochs and not self.stop_condition(self.state):
            self.model.zero_grad()

            for iteration, batch in enumerate(train_dataset_loader, 0):
                iteration_start = time()

                # Run the actual compute function
                _, _, _, self.state.last_loss = self.compute_function(batch)

                iteration_elapsed_time = time() - iteration_start

                if verbose == 1:
                    self._update_progress_bar(iteration, iteration_elapsed_time, len(train_dataset_loader), max_epochs)

            self.state.current_epoch += 1
            self._run_post_epoch_callbacks()

        print("train time %.2f" % (time() - train_start))

    def register_post_epoch_callback(self, callback):
        from callback import Callback

        if not issubclass(callback.__class__, Callback):
            raise TypeError("Argument callback should inherit from Callback.")

        self.post_epoch_callback.append(callback)

    def _run_post_epoch_callbacks(self):
        for cb in self.post_epoch_callback:
            cb(self)

    def _update_progress_bar(self, iteration, iteration_elapsed_time, train_dataset_loader_size, max_epochs):
        remaining_time_estimation = int(iteration_elapsed_time * (train_dataset_loader_size - iteration))
        print_progress(iteration + 1, train_dataset_loader_size,
                       bar_length=25,
                       prefix="epoch %d/%d" % (self.state.current_epoch + 1, max_epochs),
                       suffix="%d/%d | %.2f s/it | %s remaining | last loss %.4f " %
                              (iteration + 1, train_dataset_loader_size, iteration_elapsed_time,
                               str(datetime.timedelta(seconds=remaining_time_estimation)),
                               self.state.last_loss
                               ))


def _prepare_batch(batch, device=None, non_blocking=False):
    x, y = batch
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    x = x.to(device=device, non_blocking=non_blocking)
    y = y.to(device=device, non_blocking=non_blocking)
    return x, y


def create_default_trainer(model: nn.Module, optimizer: optim.Optimizer, criterion,
                           device=None, non_blocking=False,
                           stop_condition=NoStopping(),
                           prepare_batch=_prepare_batch,
                           output_transform=lambda x, y, y_pred, loss: (x, y, y_pred, loss.item()),
                           init_callback=None):

    if not callable(prepare_batch):
        raise TypeError("Argument prepare_batch should be a function.")
    if not callable(output_transform):
        raise TypeError("Argument output_transform should be a function.")

    if device:
        model.to(device)

    def _default_compute_function(batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return ModuleTrainer(model, optimizer,
                         compute_function=_default_compute_function,
                         stop_condition=stop_condition,
                         init_callback=init_callback)
