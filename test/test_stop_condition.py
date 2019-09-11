import unittest
import torch
from torch.utils.data import DataLoader

from pytorchtrainer import create_default_trainer
from pytorchtrainer.stop_condition import EarlyStopping
from pytorchtrainer.callback import MetricCallback
from pytorchtrainer.metric import Accuracy

from test.common import XorModule, XorDataset


class MyEarlyStopping(EarlyStopping):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

        self.MAX_EPOCHS = 1000

    def tearDown(self):
        super().tearDown()
        del self.model, self.optimizer, self.criterion, self.train_loader

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def test_early_stopping_default(self):
        early_stopping = MyEarlyStopping(patience=10)

        trainer = create_default_trainer(self.model, self.optimizer, self.criterion)
        trainer.train(self.train_loader, max_epochs=self.MAX_EPOCHS, stop_condition=early_stopping)

        self.assertTrue(early_stopping.has_been_triggered)
        self.assertTrue(early_stopping.counter == early_stopping.patience)
        self.assertNotEqual(early_stopping.best_score, float('inf'))
        self.assertTrue(trainer.state.current_epoch < self.MAX_EPOCHS)

    def test_early_stopping_accuracy(self):
        accuracy = MetricCallback(metric=Accuracy(prediction_transform=lambda x: x.round()))

        early_stopping = MyEarlyStopping(patience=50,
                                         metric=lambda state: getattr(state, accuracy.state_attribute_name),
                                         comparison_function=lambda metric, best: round(metric, 2) <= round(best, 2))

        trainer = create_default_trainer(self.model, self.optimizer, self.criterion)
        trainer.register_post_iteration_callback(accuracy)
        trainer.add_progressbar_metric("accuracy %.2f", [accuracy.state_attribute_name])

        trainer.train(self.train_loader, max_epochs=self.MAX_EPOCHS, stop_condition=early_stopping)

        self.assertTrue(early_stopping.has_been_triggered)
        self.assertTrue(early_stopping.counter == early_stopping.patience)
        self.assertNotEqual(early_stopping.best_score, accuracy.state_attribute_default_value)
        self.assertTrue(trainer.state.current_epoch < self.MAX_EPOCHS)
