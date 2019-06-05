import unittest
import torch
from torch.utils.data import DataLoader

from stop_condition import EarlyStopping
from test.common import XorModule, XorDataset
from trainer import create_default_trainer


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

    def tearDown(self):
        super().tearDown()

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def test_early_stopping(self):
        MAX_EPOCHS = 1000

        model = XorModule()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
        criterion = torch.nn.MSELoss()

        train_loader = DataLoader(XorDataset(),
                                  batch_size=1,
                                  num_workers=1
                                  )

        early_stopping = MyEarlyStopping()

        trainer = create_default_trainer(model, optimizer, criterion, stop_condition=early_stopping)
        trainer.train(train_loader, max_epochs=MAX_EPOCHS, verbose=1)

        self.assertTrue(early_stopping.has_been_triggered)
        self.assertTrue(early_stopping.counter == early_stopping.patience)
        self.assertNotEqual(early_stopping.best_score, float('inf'))
        self.assertTrue(trainer.state.current_epoch < MAX_EPOCHS)
