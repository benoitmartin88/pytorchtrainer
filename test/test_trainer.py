import os
import shutil
import unittest
import torch
from torch.utils.data import DataLoader

from test.common import XorModule, XorDataset
from trainer import create_default_trainer, ModuleTrainer

from callback.checkpoint import default_save_diretory, default_filename, SaveCheckpointCallback, LoadCheckpointCallback


class TestTrainer(unittest.TestCase):

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
        shutil.rmtree(os.path.join(default_save_diretory), ignore_errors=True)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def _evaluate_model(self, model, train_loader):
        # evaluate trained model
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(train_loader, 0):
                x, y = batch[0], batch[1]
                y_pred = model(x)
                print("x=%s, y=%s" % (x, y_pred.item()))
                self.assertEqual(int(round(y_pred.item())), int(y.item()))

    def test_xor(self):
        trainer = create_default_trainer(self.model, self.optimizer, self.criterion)
        trainer.train(self.train_loader, max_epochs=100, verbose=1)

        self._evaluate_model(self.model, self.train_loader)

    def test_checkpoint_save(self):
        trainer = create_default_trainer(self.model, self.optimizer, self.criterion)
        trainer.register_post_epoch_callback(SaveCheckpointCallback(save_every=1))
        trainer.train(self.train_loader, max_epochs=1, verbose=1)

        self.assertTrue(os.path.exists(os.path.join(default_save_diretory, default_filename)))

    def test_checkpoint_load(self):
        trainer = create_default_trainer(self.model, self.optimizer, self.criterion)
        trainer.register_post_epoch_callback(SaveCheckpointCallback(save_every=1))
        trainer.train(self.train_loader, max_epochs=1, verbose=1)

        self.assertTrue(os.path.exists(os.path.join(default_save_diretory, default_filename)))

        del trainer

        global has_callback_been_called
        has_callback_been_called = False

        # load from checkpoint
        def _callback(trainer: ModuleTrainer):
            from stop_condition.early_stopping import EarlyStopping
            trainer.stop_condition = EarlyStopping()
            trainer.optimizer.lr_scheduler = 1e-3

            global has_callback_been_called
            has_callback_been_called = True

        trainer = create_default_trainer(self.model, self.optimizer, self.criterion, init_callback=LoadCheckpointCallback(callback=_callback))

        self.assertTrue(has_callback_been_called)

        trainer.train(self.train_loader, max_epochs=2, verbose=1)

        self.assertEqual(2, trainer.state.current_epoch)
