import unittest
import torch
from torch.utils.data import DataLoader

from pytorchtrainer import create_default_trainer
from pytorchtrainer.stop_condition import EarlyStopping

from test.common import XorModule, XorDataset


class MyEarlyStopping(EarlyStopping):
    def __init__(self):
        super().__init__(patience=10)
        self.has_been_triggered = False

    def __call__(self, state):
        self.has_been_triggered = super().__call__(state)
        return self.has_been_triggered


class TestStopCondition(unittest.TestCase):

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

    def test_early_stopping(self):
        MAX_EPOCHS = 1000

        early_stopping = MyEarlyStopping()

        trainer = create_default_trainer(self.model, self.optimizer, self.criterion, stop_condition=early_stopping)
        trainer.train(self.train_loader, max_epochs=MAX_EPOCHS, verbose=1)

        self.assertTrue(early_stopping.has_been_triggered)
        self.assertTrue(early_stopping.counter == early_stopping.patience)
        self.assertNotEqual(early_stopping.best_score, float('inf'))
        self.assertTrue(trainer.state.current_epoch < MAX_EPOCHS)
