import os
import shutil
import unittest
import torch
from torch.utils.data import DataLoader

from test.common import XorModule, XorDataset
from trainer import create_default_trainer

from callback.checkpoint import SaveCheckpointCallback, default_save_diretory, default_filename


class TestTrainer(unittest.TestCase):

    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(os.path.join(default_save_diretory))

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
        model = XorModule()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = torch.nn.MSELoss()

        train_loader = DataLoader(XorDataset(),
                                  batch_size=1,
                                  num_workers=1
                                  )

        trainer = create_default_trainer(model, optimizer, criterion)
        trainer.train(model, train_loader, max_epochs=100, verbose=1)

        self._evaluate_model(model, train_loader)

    def test_checkpoint_save(self):
        model = XorModule()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = torch.nn.MSELoss()

        train_loader = DataLoader(XorDataset(),
                                  batch_size=1,
                                  num_workers=1
                                  )

        trainer = create_default_trainer(model, optimizer, criterion)

        trainer.register_post_epoch_callback(SaveCheckpointCallback(save_every=1))

        trainer.train(train_loader, max_epochs=1, verbose=1)

        self.assertTrue(os.path.exists(os.path.join(default_save_diretory, default_filename)))


"""
    def test_checkpoint_load(self):
        model = XorModule()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = torch.nn.MSELoss()

        train_loader = DataLoader(XorDataset(),
                                  batch_size=1,
                                  num_workers=1
                                  )

        trainer = create_default_trainer(model, optimizer, criterion)

        trainer.register_post_epoch_callback(SaveCheckpointCallback(save_every=1))

        trainer.train(train_loader, max_epochs=1, verbose=1)

        self.assertTrue(os.path.exists(os.path.join(default_save_diretory, default_filename)))
"""
