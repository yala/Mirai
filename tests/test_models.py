import unittest
import sys
import os
import torch
import torch.nn as nn

# append module root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onconet.models.factory as mf
import onconet.models.default_models


class Args():
    pass


class TestModels(unittest.TestCase):
    def setUp(self):
        self.args = Args()
        self.args.weight_decay = 5e-5
        self.args.lr = 0.001
        self.model = nn.Linear(10, 2)

    def tearDown(self):
        self.args = None
        self.model = None

    def test_get_existing_optimizers(self):
        args = self.args
        optimizers = [
            ('adam', torch.optim.Adam),
        ]
        for optimizer, optim_type in optimizers:
            args.optimizer = optimizer
            optim = mf.get_optimizer(self.model, args)
            self.assertIsInstance(optim, optim_type)

    def test_non_existing_optimizers(self):
        args = self.args
        optimizers = [
            None,
            'yala',
            5,
        ]
        for optimizer in optimizers:
            args.optimizer = optimizer
            with self.assertRaises(Exception) as context:
                mf.get_optimizer(self.model, args)

            self.assertTrue(
                'Optimizer {} not supported!'.format(optimizer) in str(
                    context.exception))


if __name__ == '__main__':
    unittest.main()
