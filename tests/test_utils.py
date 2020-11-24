import unittest
import sys
import os
import shutil
import argparse
import torch
import tempfile
import torchvision.models as models
import numpy as np

# append module root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onconet.utils.parsing as parsing
import onconet.utils.generic as generic
import onconet.train.state_keeper as state
import datetime

class Test_parse_transformers(unittest.TestCase):
    def setUp(self):
        self.err_msg = "Name of transformer or one of its arguments cant be empty\n\
                  Use 'name/arg1=value/arg2=value' format"

    def tearDown(self):
        pass

    def test_parse_transformers(self):
        raw_strings = [
            # 1
            (['my_transformer'], [('my_transformer', {})]),
            # 2
            (['t1', 't2'], [('t1', {}), ('t2', {})]),
            # 3
            ([], []),
            # 4
            (['t1/a=v'], [('t1', {
                'a': 'v'
            })]),
            # 5
            (['t1/a=v1/b=v2'], [('t1', {
                'a': 'v1',
                'b': 'v2',
            })]),
            # 6
            (['t1/a=v1/b=v2', 't2/c=v3'], [('t1', {
                'a': 'v1',
                'b': 'v2',
            }), ('t2', {
                'c': 'v3'
            })]),
        ]

        for raw_transformers, expected in raw_strings:
            output = parsing.parse_transformers(raw_transformers)
            self.assertEqual(output, expected)

    def test_parse_transformers_unvalid(self):
        raw_strings = [['/a=4'], ['t/'], ['t/=5']]

        for raw_transformers in raw_strings:
            with self.assertRaises(Exception) as context:
                parsing.parse_transformers(raw_transformers)

            self.assertTrue(self.err_msg in str(context.exception))


class Test_parse_dispatcher_config(unittest.TestCase):
    def setUp(self):
        self.err_msg = "Flag {} has an invalid list of values: {}. Length of list must be >=1"

    def tearDown(self):
        pass

    def test_parse_dispatcher_config(self):
        raw_strings = [
            # true flag
            ({
                'true_flag': [True]
            }, [" --true_flag"]),
            # flase flag
            ({
                'false_flag': [False]
            }, [""]),
            # flase flag followed by true flag
            ({
                'false_flag': [False],
                'true_flag': [True]
            }, [" --true_flag"]),
            # Different options
            ({
                'flag': ["opt1", "opt2"]
            }, [" --flag opt1", " --flag opt2"]),
            # val
            ({
                "val_flag": [5]
            }, [" --val_flag 5"]),
            # list of vals
            ({
                "list_flag": [['t1/a=v', 't2/b=w']]
            }, [" --list_flag t1/a=v t2/b=w"])
        ]

        for config, expected in raw_strings:
            wrapped_config = {"search_space": config}
            output, _ = parsing.parse_dispatcher_config(wrapped_config)
            self.assertCountEqual(output, expected)

    def test_experimental_axies(self):
        raw_strings = [
            ({
                'experimented_flag': [True, False],
                'const_flag': [False]
            }, "experimented_flag"),
        ]

        for config, expected in raw_strings:
            wrapped_config = {"search_space": config}
            _, output = parsing.parse_dispatcher_config(wrapped_config)
            self.assertEqual(output[0], expected)

    def test_invalid_config(self):
        raw_strings = [
            ({
                'flag1': [True, False],
                'flag2': []
            }, 'flag2'),
            ({
                'flag1': []
            }, 'flag1'),
            ({
                'flag1': [True],
                'flag2': "value",
            }, 'flag2'),
        ]

        for config, flag in raw_strings:
            wrapped_config = {"search_space": config}
            with self.assertRaises(Exception) as context:
                parsing.parse_dispatcher_config(wrapped_config)

            self.assertTrue(
                self.err_msg.format(flag,
                                    config[flag]) in str(context.exception))

class Test_generic_utils(unittest.TestCase):
    def setUp(self):
        self.err_msg = "Date string not valid!"

    def tearDown(self):
        pass

    def test_normalize_dictionary(self):
        examples = [
            # several int val
            ({
                'a': 1,
                'b':1
            }, {'a':.5, 'b':.5}),
            # several float val
            ({
                'a': 1.5,
                'b':.5
            }, {'a':.75, 'b':.25}),
            # single val
            ({
                'a': 1.5,
            }, {'a':1.0})
            ]


        for inp, expected in examples:
            output = generic.normalize_dictionary(inp)
            self.assertDictEqual(output, expected)

    def test_iso_str_to_datetime_obj(self):

        date_examples = [
        # random date
        ('1995-02-26T00:00:00',
         datetime.datetime(1995,2,26)),
        ('2205-12-10T00:00:00',
         datetime.datetime(2205,12,10))
        ]

        exception_example = '199b-04-20T22:23:00'

        for inp, expected in date_examples:
            output = generic.iso_str_to_datetime_obj(inp)
            self.assertTrue(output == expected)

        with self.assertRaises(Exception) as context:
            generic.iso_str_to_datetime_obj(exception_example)

        self.assertTrue(self.err_msg in str(context.exception))


class Test_hashing(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser(description='Test Parser')
        self.parser.add_argument('--firstname', default='John')
        self.parser.add_argument('--lastname', default='Doe')
        self.parser.add_argument('--age', default=10)
        self.parser.add_argument('--siblings', default=["alice", "bob"])
    
    def tearDown(self):
        pass

    def test_different_arguments(self):
        args, unknown = self.parser.parse_known_args()
        hash1 = state.get_identifier(args)

        args = self.parser.parse_args(['--firstname', 'Ben'])
        hash2 = state.get_identifier(args)

        self.assertNotEqual(hash1, hash2)

        args = self.parser.parse_args(['--lastname', 'Bittidle', '--age', '12'])
        hash3 = state.get_identifier(args)

        self.assertNotEqual(hash1, hash3)
        self.assertNotEqual(hash2, hash3)

    def test_same_arguments(self):
        args, unknown = self.parser.parse_known_args()

        hash1 = state.get_identifier(args)

        args = self.parser.parse_args(['--firstname', 'John'])
        hash2 = state.get_identifier(args)

        self.assertEqual(hash1, hash2)

class Test_state_keeping(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser(description='Test Parser')
        self.parser.add_argument('--firstname', default='John')
        self.parser.add_argument('--lastname', default='Doe')
        self.parser.add_argument('--age', default=10)
        self.parser.add_argument('--siblings', default=["alice", "bob"])
        self.parser.add_argument('--save_dir', default=tempfile.mkdtemp())
        self.epoch = 10
        self.lr = 0.001
        self.epoch_stats = {}
        self.model = models.resnet18()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.01, momentum=0.9)
    
    def tearDown(self):
        args, unknown = self.parser.parse_known_args()
        shutil.rmtree(args.save_dir)
        pass

    def test_simple_model(self):
        args, unknown = self.parser.parse_known_args()
        state_keeper = state.StateKeeper(args)

        model_dict = self.model.state_dict()
        optimizer_dict = self.optimizer.state_dict()

        state_keeper.save(self.model, self.optimizer, self.epoch, self.lr, self.epoch_stats)
        new_model, new_optimizer_state, new_epoch, new_lr, _ = state_keeper.load()

        for key in model_dict.keys():
            self.assertTrue(np.array_equal(model_dict[key].numpy(), new_model.state_dict()[key].numpy()))
        self.assertEqual(optimizer_dict, new_optimizer_state)
        self.assertEqual(self.epoch, new_epoch)
        self.assertEqual(self.lr, new_lr)


if __name__ == '__main__':
    unittest.main()
