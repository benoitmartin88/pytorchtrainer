import os
import shutil
import unittest
import torch
from torch.utils.data import DataLoader

from pytorchtrainer import create_default_trainer
from pytorchtrainer.callback import ValidationCallback, SaveBestCheckpointCallback, checkpoint, file_writer
from pytorchtrainer.metric import TorchLoss

from test.common import XorModule, XorDataset


class MyValidationCallback(ValidationCallback):
    def __init__(self, dataset_loader, metric, validate_every):
        super().__init__(dataset_loader, metric, validate_every=validate_every)
        self.has_been_called = False

    def __call__(self, trainer):
        super().__call__(trainer)
        self.has_been_called = True


class MySaveBestCheckpointCallback(SaveBestCheckpointCallback):
    def __init__(self, state_metric_name, saves_to_keep):
        super().__init__(state_metric_name, saves_to_keep=saves_to_keep)
        self.has_been_called = False

    def __call__(self, trainer):
        super().__call__(trainer)
        self.has_been_called = True


class TestCallback(unittest.TestCase):

    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = XorModule()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.criterion = torch.nn.MSELoss()

        self.train_loader = DataLoader(XorDataset(),
                                       batch_size=1,
                                       num_workers=1
                                       )

    def tearDown(self):
        super().tearDown()
        del self.model, self.optimizer, self.criterion, self.train_loader

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(os.path.join(checkpoint.default_save_diretory), ignore_errors=True)
        shutil.rmtree(os.path.join(file_writer.default_save_directory), ignore_errors=True)

    def test_frequency_error(self):
        from pytorchtrainer.callback import Callback
        self.assertRaises(ValueError, Callback, frequency=-1)
        self.assertRaises(NotImplementedError, Callback(frequency=0), trainer=None)

    def test_validation(self):
        validation_callback = MyValidationCallback(self.train_loader, TorchLoss(self.criterion), validate_every=1)

        trainer = create_default_trainer(self.model, self.optimizer, self.criterion)
        trainer.register_post_iteration_callback(validation_callback)
        trainer.register_post_iteration_callback(file_writer.CsvWriter(save_every=1,
                                                                       extra_header=[validation_callback.state_attribute_name],
                                                                       callback=lambda state: [state.get(validation_callback.state_attribute_name)]))
        trainer.register_post_epoch_callback(validation_callback)
        trainer.train(self.train_loader, max_epochs=5)

        self.assertTrue(validation_callback.has_been_called)
        self.assertTrue(trainer.state.last_validation_loss != float('inf'))

    def test_save_best(self):
        validation_callback = ValidationCallback(self.train_loader, TorchLoss(self.criterion), validate_every=1)
        callback = MySaveBestCheckpointCallback(validation_callback.state_attribute_name, saves_to_keep=1)

        trainer = create_default_trainer(self.model, self.optimizer, self.criterion)
        trainer.register_post_epoch_callback(validation_callback)
        trainer.register_post_epoch_callback(callback)
        trainer.train(self.train_loader, max_epochs=10)

        self.assertTrue(callback.has_been_called)

