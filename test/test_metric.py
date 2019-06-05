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

    def test_mean_absolute_error(self):
        from metric.mean_absolute_error import MeanAbsoluteError
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