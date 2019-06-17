import os
import shutil
import unittest
import torch
from torch.utils.data import DataLoader

from pytorchtrainer import create_default_trainer, ModuleTrainer
from pytorchtrainer.callback import checkpoint, file_writer, MetricCallback, ValidationCallback
from pytorchtrainer.metric import Accuracy, Loss

from test.common import XorModule, XorDataset


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

        self.prediction_transform = lambda x: int(round(x.item()))

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(os.path.join(checkpoint.default_save_diretory), ignore_errors=True)
        shutil.rmtree(os.path.join(file_writer.default_save_directory), ignore_errors=True)
        del self.model, self.optimizer, self.criterion, self.train_loader

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
                self.assertEqual(self.prediction_transform(y_pred), int(y.item()))

    def test_xor(self):
        trainer = create_default_trainer(self.model, self.optimizer, self.criterion)
        trainer.train(self.train_loader, max_epochs=100, verbose=1)

        self._evaluate_model(self.model, self.train_loader)

    def test_dtype(self):
        for dtype in [torch.float32, torch.float64]:
            trainer = create_default_trainer(self.model, self.optimizer, self.criterion, dtype=dtype)
            trainer.train(self.train_loader, max_epochs=5, verbose=1)
            del trainer

    def test_checkpoint_save(self):
        trainer = create_default_trainer(self.model, self.optimizer, self.criterion)
        trainer.register_post_epoch_callback(checkpoint.SaveCheckpointCallback(save_every=1))
        trainer.train(self.train_loader, max_epochs=1, verbose=1)

        self.assertTrue(os.path.exists(os.path.join(checkpoint.default_save_diretory, checkpoint.default_filename)))

    def test_checkpoint_load(self):
        trainer = create_default_trainer(self.model, self.optimizer, self.criterion)
        trainer.register_post_epoch_callback(checkpoint.SaveCheckpointCallback(save_every=1))
        trainer.train(self.train_loader, max_epochs=1, verbose=1)

        self.assertTrue(os.path.exists(os.path.join(checkpoint.default_save_diretory, checkpoint.default_filename)))

        del trainer

        global has_callback_been_called
        has_callback_been_called = False

        # load from checkpoint
        def _callback(trainer: ModuleTrainer):
            from pytorchtrainer.stop_condition import EarlyStopping
            trainer.stop_condition = EarlyStopping()
            trainer.optimizer.lr_scheduler = 1e-3

            global has_callback_been_called
            has_callback_been_called = True

        trainer = create_default_trainer(self.model, self.optimizer, self.criterion,
                                         init_callback=checkpoint.LoadCheckpointCallback(callback=_callback))

        self.assertTrue(has_callback_been_called)

        trainer.train(self.train_loader, max_epochs=2, verbose=1)

        self.assertEqual(2, trainer.state.current_epoch)

    def test_log_save(self):
        writer = file_writer.CsvWriter(save_every=1, extra_header=['test'], callback=lambda trainer: [42])

        trainer = create_default_trainer(self.model, self.optimizer, self.criterion)
        trainer.register_post_epoch_callback(writer)
        trainer.train(self.train_loader, max_epochs=10, verbose=1)

        self.assertTrue(os.path.exists(writer.log_file_path))

    def test_progressbar_extra_metric(self):
        trainer = create_default_trainer(self.model, self.optimizer, self.criterion)

        # Validation callback
        trainer.register_post_epoch_callback(ValidationCallback(self.train_loader, metric=Loss(self.criterion), validate_every=1))

        # Accuracy
        trainer.register_post_iteration_callback(MetricCallback(
            metric=Accuracy(prediction_transform=self.prediction_transform),
            frequency=1))

        trainer.add_progressbar_metric("validation loss %.4f | accuracy %.2f", ["last_validation_loss", "accuracy"])

        trainer.train(self.train_loader, max_epochs=10, verbose=1)
