import collections
import datetime
from time import time
import signal

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

from .metric import Metric
from .callback import Callback
from .stop_condition import NoStopping
from .utils import print_progress, batch_to_tensor


class State(object):
    current_epoch = 0
    current_iteration = 1
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
    """
    Runs a training configurable training loop over a model.
    """
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, train_function, evaluate_function,
                 prepare_batch_function=batch_to_tensor, init_callback=None, verbose=1):
        """
        :param model: PyTorch model (nn.Module) that is to be trained.
        :param optimizer: PyTorch optimizer (optim.Optimizer) that will be used.
        :param train_function: The main training function that will be run within the epoch and iteration loops.
        :param evaluate_function: The evaluation function that will be used.
        :param init_callback: A function that will be run at the end of the trainer's constructor. Can be used to load/resume a previously trained model.
        :param verbose: Currently only used to show the progressbar. By default this is set to 1.
        """
        if not callable(train_function):
            raise TypeError("Argument train_function should be a function.")
        if not callable(evaluate_function):
            raise TypeError("Argument evaluate_function should be a function.")
        if not callable(prepare_batch_function):
            raise TypeError("Argument prepare_batch_function should be a function.")

        signal.signal(signal.SIGINT, self.__graceful_exit)

        self.state = State()
        self.model = model
        self.optimizer = optimizer
        self.train_function = train_function
        self.evaluate_function = evaluate_function
        self.prepare_batch_function = prepare_batch_function

        self.__post_iteration_callback = []
        self.__post_epoch_callback = []

        self.extra_progressbar_metrics = ()

        if init_callback is not None:
            if not callable(init_callback):
                raise TypeError("Argument post_init_callback should be a function.")

            init_callback(self)

        self.verbose = verbose
        if self.verbose == 1:
            print(model)

    def train(self, train_dataloader: torch.utils.data.DataLoader, max_epochs=100, stop_condition=NoStopping()):
        """
        Train the model using a given data-loader.
        Training will be stopped when `max_epochs` has been reached or if `stop_condition` returns True.
        :param train_dataloader: PyTorch DataLoader that will be used to load training batches.
        :param max_epochs: Maximum number of epochs to run.
        :param stop_condition: Stop training loop based on a defined condition. By default training will not be stopped.
        """
        if not callable(stop_condition):
            raise TypeError("Argument stop_condition should be a function.")

        self.model.train()  # set the module to training mode

        train_start = time()
        while self.state.current_epoch < max_epochs and not stop_condition(self.state):
            self.model.zero_grad()

            for self.state.current_iteration, batch in enumerate(train_dataloader):
                iteration_start = time()
                self.state.current_iteration += 1

                # Run the actual compute function
                self.state.last_x, self.state.last_y, self.state.last_y_pred, self.state.last_train_loss = self.train_function(batch)

                self.__run_post_iteration_callbacks()

                iteration_elapsed_time = time() - iteration_start

                if self.verbose == 1:
                    self.__update_progress_bar(iteration_elapsed_time, len(train_dataloader), max_epochs)

            self.state.current_epoch += 1
            self.__run_post_epoch_callbacks()

        print("train time %.2f" % (time() - train_start))

    def evaluate(self, dataloader: torch.utils.data.DataLoader, metric: Metric):
        """
        Evaluate a model.
        :param dataloader:  PyTorch DataLoader that will be used to load the evaluation batches.
        :param metric:  Metric object that will be used to compute the evaluation metric.
        :return:    Computed metric after having evaluated the model across the entire dataloader.
        """

        previous_training_flag = self.model.training

        self.model.eval()
        metric.reset()

        with torch.no_grad():
            for batch in dataloader:
                _, y, y_pred, _ = self.evaluate_function(batch)
                metric.step(y, y_pred)

        self.model.train(previous_training_flag)
        return metric.compute()

    def add_progressbar_metric(self, format: str, callbacks: list):
        if not isinstance(format, str) or not isinstance(callbacks, list):
            raise TypeError("format should be string (eg: '%.2f') and metric_state_names should be a list of metric names (eg: 'accuracy')")

        for callback in callbacks:
            if not isinstance(callback, Callback):
                raise TypeError("callbacks argument should only contain Callback instances.")

            assert hasattr(callback, "state_attribute_name")

            if callback.state_attribute_name is None:
                raise RuntimeError("callback %s cannot be used. It does not store any data to the trainer's state." % callback.state_attribute_name)

        self.extra_progressbar_metrics = (format, callbacks)

    def register_post_iteration_callback(self, callback: Callback, frequency: int = 1):
        """
        Add a callback function that will be run after each iteration
        :param callback: Function that will be run
        :param frequency: Frequency at which the callback will be called
        """
        if frequency < 0:
            raise ValueError("frequency argument should be positive.")

        callback.set_trainer(self)  # This is done here to simplify the API by hiding this from the user
        self.__post_iteration_callback.append((frequency, callback))

    def register_post_epoch_callback(self, callback: Callback, frequency: int = 1):
        """
        Add a callback function that will be run after each epoch
        :param callback: Function that will be run
        :param frequency: Frequency at which the callback will be called
        """
        if frequency < 0:
            raise ValueError("frequency argument should be positive.")

        callback.set_trainer(self)    # This is done here to simplify the API by hiding this from the user
        self.__post_epoch_callback.append((frequency, callback))

    def __graceful_exit(self, signum, frame):
        print("Sig %s caught. Graceful exit has been called. Currently running epoch will be finished." % signum)
        self.stop_condition = lambda state: True

    def __run_post_iteration_callbacks(self):
        for frequency, callback in self.__post_iteration_callback:
            if frequency != 0 and self.state.current_iteration % frequency == 0:
                callback(self)

    def __run_post_epoch_callbacks(self):
        for frequency, callback in self.__post_epoch_callback:
            if frequency != 0 and self.state.current_epoch % frequency == 0:
                callback(self)

    def __update_progress_bar(self, iteration_elapsed_time, train_dataloader_size, max_epochs):
        remaining_time_estimation = int(iteration_elapsed_time * (train_dataloader_size - self.state.current_iteration))

        if len(self.extra_progressbar_metrics) > 0:
            suffix = ("%d/%d | %.2f s/it | %s remaining | train loss %.4f | " + self.extra_progressbar_metrics[0]) % \
                     (self.state.current_iteration, train_dataloader_size, iteration_elapsed_time,
                      str(datetime.timedelta(seconds=remaining_time_estimation)), self.state.last_train_loss,
                      *[getattr(self.state, a.state_attribute_name) for a in self.extra_progressbar_metrics[1]])
        else:
            suffix = "%d/%d | %.2f s/it | %s remaining | train loss %.4f" % \
                     (self.state.current_iteration, train_dataloader_size, iteration_elapsed_time,
                      str(datetime.timedelta(seconds=remaining_time_estimation)), self.state.last_train_loss)

        print_progress(self.state.current_iteration, train_dataloader_size,
                       bar_length=25,
                       prefix="epoch %d/%d" % (self.state.current_epoch + 1, max_epochs),
                       suffix=suffix)


def create_default_trainer(model: nn.Module, optimizer: optim.Optimizer, criterion,
                           device=None, dtype=None, non_blocking=False,
                           prepare_batch_function=batch_to_tensor,
                           loss_transform_function=lambda criterion, y_preds, y: criterion(y_preds, y),
                           output_transform_function=lambda x, y, y_pred, loss: (x, y, y_pred, loss.item() if loss is not None else None),
                           init_callback=None,
                           verbose=1):
    """
    Helper method that returns an instance of `ModuleTrainer`. This is helpful as it provides a default training function.
    Note: The optimizer's state will be reset when calling the method.
    :param model: Model to train.
    :param optimizer: Optimizer that will be used through-out the training.
    :param criterion: Loss function to optimize.
    :param device: device to use. eg: 'cpu' (default) or 'cuda'.
    :param dtype: Passed to `prepare_batch` argument to change the model's data type. eg: `torch.float32` or `torch.float64`.
    :param non_blocking: Passed to `prepare_batch`.
    :param prepare_batch_function: Function that prepares a batch. This should return a `torch.Tensor`.
    :param loss_transform_function: Optionally transform the loss function's output. Can be useful with multi-output models.
    :param output_transform_function: Optionally transform the `x`, `y`, `y_prediction` and `loss`. Can be useful with multi-output models.
    :param init_callback: Passed to `ModuleTrainer`'s constructor.
    :param verbose: Currently only used to show the progressbar. By default this is set to 1.
    :return: An instance of `ModuleTrainer`.
    """

    if not callable(prepare_batch_function):
        raise TypeError("Argument prepare_batch_function should be a function.")
    if not callable(output_transform_function):
        raise TypeError("Argument output_transform should be a function.")

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    if dtype:
        model.to(dtype=dtype)

    # reset optimizer state
    optimizer.state = collections.defaultdict(dict)

    def _default_train_function(batch):
        model.train()
        optimizer.zero_grad()
        x, y, model_args = prepare_batch_function(batch, device=device, dtype=dtype, non_blocking=non_blocking)
        y_pred = model(x, **model_args)
        loss = loss_transform_function(criterion, y_pred, y)
        loss.backward()
        optimizer.step()

        # does y_pred need detach() ?
        return output_transform_function(x, y, y_pred, loss.detach())

    def _default_evaluate_function(batch):
        x, y, model_args = prepare_batch_function(batch, device=device, dtype=dtype, non_blocking=non_blocking)
        y_pred = model(x, **model_args)

        # does y_pred need detach() ?
        return output_transform_function(x, y, y_pred, None)

    return ModuleTrainer(model, optimizer,
                         train_function=_default_train_function,
                         evaluate_function=_default_evaluate_function,
                         prepare_batch_function=prepare_batch_function,
                         init_callback=init_callback,
                         verbose=verbose)
