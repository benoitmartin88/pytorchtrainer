import unittest
import torch
from torch.utils.data import DataLoader

from callback import ValidationCallback
from metric import Loss
from test.common import XorModule, XorDataset
from trainer import create_default_trainer


class MyValidationCallback(ValidationCallback):
    def __init__(self, dataset_loader, metric, validate_every):
        super().__init__(dataset_loader, metric, validate_every=validate_every)
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

    def test_validation(self):
        validation_callback = MyValidationCallback(self.train_loader, Loss(self.criterion), validate_every=1)

        trainer = create_default_trainer(self.model, self.optimizer, self.criterion)
        trainer.register_post_iteration_callback(validation_callback)
        trainer.register_post_epoch_callback(validation_callback)
        trainer.train(self.train_loader, max_epochs=5, verbose=1)

        self.assertTrue(validation_callback.has_been_called)
        self.assertTrue(trainer.state.last_validation_loss != float('inf'))
