import unittest
import torch


class TestMetric(unittest.TestCase):

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

    def test_errors(self):
        from pytorchtrainer.metric import Metric
        metric = Metric("")
        self.assertRaises(NotImplementedError, metric.step, None, None)
        self.assertRaises(NotImplementedError, metric.compute)
        self.assertRaises(NotImplementedError, metric.reset)

    def test_accuracy(self):
        from pytorchtrainer.metric import Accuracy
        accuracy = Accuracy()
        accuracy.step(torch.tensor([[1]]), torch.tensor([[1]]))

        self.assertEqual(1, accuracy._total)
        self.assertEqual(1, accuracy._total_correct)
        self.assertEqual(1.0, accuracy.compute())

        accuracy.reset()
        self.assertEqual(0, accuracy._total)
        self.assertEqual(0, accuracy._total_correct)

        accuracy.step(torch.tensor([[1]]), torch.tensor([[2]]))
        self.assertEqual(1, accuracy._total)
        self.assertEqual(0, accuracy._total_correct)
        self.assertEqual(0.0, accuracy.compute())

        accuracy.step(torch.tensor([[2]]), torch.tensor([[2]]))
        self.assertEqual(2, accuracy._total)
        self.assertEqual(1, accuracy._total_correct)
        self.assertEqual(0.5, accuracy.compute())

        accuracy.step(torch.tensor([[1], [1]]), torch.tensor([[1], [3]]))
        self.assertEqual(4, accuracy._total)
        self.assertEqual(2, accuracy._total_correct)
        self.assertEqual(0.5, accuracy.compute())

    def test_mean_absolute_error(self):
        from pytorchtrainer.metric import MeanAbsoluteError
        mea = MeanAbsoluteError()
        mea.step(torch.tensor([[1]]), torch.tensor([[1]]))

        self.assertEqual(1, mea._total)
        self.assertEqual(0, mea.compute())

        mea.reset()
        self.assertEqual(0, mea._absolute_error_sum)
        self.assertEqual(0, mea._total)

        mea.step(torch.tensor([[1]]), torch.tensor([[2]]))

        self.assertEqual(1, mea._total)
        self.assertEqual(1, mea.compute())

        mea.step(torch.tensor([[1]]), torch.tensor([[3]]))

        self.assertEqual(2, mea._total)
        self.assertEqual(1.5, mea.compute())

        mea.step(torch.tensor([[1], [1]]), torch.tensor([[2], [1]]))

        self.assertEqual(4, mea._total)
        self.assertEqual(1.0, mea.compute())

    def test_torchloss_l1(self):
        from pytorchtrainer.metric import TorchLoss
        import torch
        l1 = TorchLoss(torch.nn.L1Loss())
        l1.step(torch.tensor([[1.]]), torch.tensor([[1.]]))

        self.assertEqual(0, l1.compute())

        l1.reset()
        self.assertEqual(0, l1._loss_sum)
        self.assertEqual(0, l1._total)

        l1.step(torch.tensor([[1.]]), torch.tensor([[2.]]))

        self.assertEqual(1, l1.compute())

        l1.reset()
        l1.step(torch.tensor([[1.]]), torch.tensor([[3.]]))

        self.assertEqual(2, l1.compute())

        l1.reset()
        l1.step(torch.tensor([[1.], [1.]]), torch.tensor([[2.], [1.]]))
        l1.step(torch.tensor([[1.], [1.]]), torch.tensor([[2.], [2.]]))

        self.assertEqual(0.75, l1.compute())
