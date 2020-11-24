import unittest
import sys
import os
import torch
import torch.nn as nn
import tempfile
import shutil
from PIL import Image

# append module root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onconet.datasets.loader.image as image_loader
from onconet.transformers.abstract import Abstract_transformer
import onconet.utils.parsing as parsing
import onconet

ERROR_MSG = 'Test {} failed: {} != {}'

white_image = Image.new("RGB", (1, 1), "white")
black_image = Image.new("RGB", (1, 1), "black")
blue_image = Image.new("RGB", (1, 1), "blue")
red_image = Image.new("RGB", (1, 1), "red")

class cachable_transformer(Abstract_transformer):
    def __init__(self, name, *keys):
        super(cachable_transformer, self).__init__()
        self.name = name
        self.set_cachable(*keys)

    def __call__(self, img, additional):
        return white_image


class noncachable_transformer(Abstract_transformer):
    def __init__(self, name, *keys):
        super(noncachable_transformer, self).__init__()
        self.name = name

    def __call__(self, img, additional):
        return black_image


def get_test_transformers():
    return {'c0' : cachable_transformer('c0'),
            'c1' : cachable_transformer('c1', 'k1'),
            'c2' : cachable_transformer('c2', 'k1','k2'),
            'nc' : noncachable_transformer('nc')}


class TestSplitting(unittest.TestCase):
    def setUp(self):
        self.transformers = get_test_transformers()

    def tearDown(self):
        self.transformers = None

    def test_split_transformers_by_cache(self):
       for case in [{
                     'name': 'one cachable transformer',
                     'transformers': [self.transformers['c1']],
                     'expected': [('@c1#k1', []), ('default/', [self.transformers['c1']])]
                     },
                     {
                     'name': 'two cachable transformers',
                     'transformers': [self.transformers['c1'], self.transformers['c2']],
                     'expected': [('@c1#k1@c2#k1#k2', []), ('@c1#k1', [self.transformers['c2']]), ('default/', [self.transformers['c1'], self.transformers['c2']])]
                     },
                     {
                     'name': 'no transformers',
                     'transformers': [],
                     'expected': [('default/', [])]
                     },
                     {
                     'name': 'no cachable transformers',
                     'transformers': [self.transformers['nc']],
                     'expected': [('default/', [self.transformers['nc']])]
                     },
                     {
                     'name': 'no cachable after cachable transformers',
                     'transformers': [self.transformers['c1'], self.transformers['nc']],
                     'expected': [('@c1#k1', [self.transformers['nc']]), ('default/', [self.transformers['c1'], self.transformers['nc']])]
                     },
                     {
                     'name': 'cachable after non cachable transformers',
                     'transformers': [self.transformers['nc'], self.transformers['c1']],
                     'expected': [('default/', [self.transformers['nc'], self.transformers['c1']])]
                     },
                     ]:

            transformers = case['transformers']
            split_transformers = image_loader.split_transformers_by_cache(transformers)
            self.assertEqual(split_transformers, case['expected'], ERROR_MSG.format(case['name'], split_transformers, case['expected']))


class Test_image_loader(unittest.TestCase):
    def setUp(self):
        self.cache_path = tempfile.mkdtemp()
        self.transformers = get_test_transformers()

    def tearDown(self):
        shutil.rmtree(self.cache_path)
        self.transformers = None

    def test_loads_from_cache(self):
        transformers = [self.transformers['c1']]
        loader = image_loader.image_loader(self.cache_path, transformers)
        image_path = '/some/test'

        # Trick loader by storing specific image in cache
        key = transformers[0].caching_keys()
        loader.cache.add(image_path, key, blue_image)

        # Load image
        output = loader.get_image(image_path, None)
        self.assertEqual(output.getdata()[0], blue_image.getdata()[0])

    def test_adds_to_cache(self):
        transformers = [self.transformers['c1']]
        loader = image_loader.image_loader(self.cache_path, transformers)

        # save some test image (cached dir is used only for convinient as tmp)
        image_path = self.cache_path + 'test.png'
        blue_image.save(image_path)

        # Load image
        output = loader.get_image(image_path, None)
        self.assertEqual(output.getdata()[0], white_image.getdata()[0])

        # Validate that correct images were cached
        key = transformers[0].caching_keys()
        default_image = loader.cache.get(image_path, 'default/')
        self.assertEqual(default_image.getdata()[0], blue_image.getdata()[0])

        c1_image = loader.cache.get(image_path, key)
        self.assertEqual(c1_image.getdata()[0], white_image.getdata()[0])

    def test_non_cachable(self):
        transformers = [self.transformers['nc'], self.transformers['c1']]
        loader = image_loader.image_loader(self.cache_path, transformers)

        # save some test image
        image_path = self.cache_path + 'test.png'
        blue_image.save(image_path)

        # Load image
        output = loader.get_image(image_path, None)
        self.assertEqual(output.getdata()[0], white_image.getdata()[0])

        # Validate that default image was cached
        default_image = loader.cache.get(image_path, 'default/')
        self.assertEqual(default_image.getdata()[0], blue_image.getdata()[0])

        # Validate that 'nc' wasn't cached
        key = transformers[0].caching_keys()
        self.assertFalse(loader.cache.exists(image_path, key))

        # Validate that 'c1' wasn't cached because it is after a non cachable
        key = transformers[1].caching_keys()
        self.assertFalse(loader.cache.exists(image_path, key))


if __name__ == '__main__':
    unittest.main()
