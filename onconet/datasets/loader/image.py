import torch
import torchvision
from PIL import Image, ImageFile
import os
import sys
import os.path
import warnings
from onconet.utils.generic import md5
from onconet.transformers.basic import ComposeTrans
ImageFile.LOAD_TRUNCATED_IMAGES = True


CACHED_FILES_EXT = '.png'
DEFAULT_CACHE_DIR = 'default/'

CORUPTED_FILE_ERR = 'WARNING! Error processing file from cache - removed file from cache. Error: {}'


def split_transformers_by_cache(transformers):
    '''
    Given a list of transformers, returns a list of tuples. Each tuple
    contains a caching key of the transformers up to the spiltting point,
    and a list of transformers that should be applied afterwards.

    split_transformers will contain all possible splits by cachable transformers,
    ordered from latest possible one to the former ones.
    The last tuple will have all transformers.

    Note - splitting will be done for indexes that all transformers up to them are
    cachable.
    '''
    # list of (cache key, post transformers)
    split_transformers = []
    split_transformers.append((DEFAULT_CACHE_DIR, transformers))
    all_prev_cachable = True
    key = ''
    for ind, trans in enumerate(transformers):

        # check trans.cachable() first separately to save run time
        if not all_prev_cachable or not trans.cachable():
            all_prev_cachable = False
        else:
            key += trans.caching_keys()
            post_transformers = transformers[
                ind + 1:] if ind < len(transformers) else []
            split_transformers.append((key, post_transformers))

    return list(reversed(split_transformers))


def apply_transformers_and_cache(image,
                                 additional,
                                 img_path,
                                 transformers,
                                 cache,
                                 cache_full_size=False,
                                 base_key=''):
    '''
    Loads the image by its absolute path and apply the transformers one
    by one (similar to what the composed one is doing).  All first cachable
    transformer's output is cached (until reaching a non cachable one).
    '''
    if cache_full_size:
        cache.add(img_path, DEFAULT_CACHE_DIR, image)

    all_prev_cachable = True
    key = base_key
    for ind, trans in enumerate(transformers):
        image = trans(image, additional)
        if not all_prev_cachable or not trans.cachable():
            all_prev_cachable = False
        else:
            key += trans.caching_keys()
            cache.add(img_path, key, image)

    return image


class cache():
    def __init__(self, path, extension=CACHED_FILES_EXT):
        if not os.path.exists(path):
            os.makedirs(path)

        self.cache_dir = path
        self.files_extension = extension

    def _file_dir(self, attr_key):
        return os.path.join(self.cache_dir, attr_key)

    def _file_path(self, attr_key, hashed_key):
        return os.path.join(self.cache_dir, attr_key, hashed_key +
                                   self.files_extension)

    def exists(self, image_path, attr_key):
        hashed_key = md5(image_path)
        return os.path.isfile(self._file_path(attr_key, hashed_key))

    def get(self, image_path, attr_key):
        hashed_key = md5(image_path)
        return Image.open(self._file_path(attr_key, hashed_key))

    def add(self, image_path, attr_key, image):
        hashed_key = md5(image_path)
        file_dir = self._file_dir(attr_key)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        image.save(self._file_path(attr_key, hashed_key))

    def rem(self, image_path, attr_key):
        hashed_key = md5(image_path)
        try:
            os.remove(self._file_path(attr_key, hashed_key))
        # Don't raise error if file not exists.
        except OSError:
            pass


class image_loader():
    def __init__(self, cache_path, transformers):
        self.transformers = transformers

        if cache_path is not None:
            self.use_cache = True
            self.cache = cache(cache_path)
            self.split_transformers = split_transformers_by_cache(transformers)
        else:
            self.use_cache = False
            self.composed_all_transformers = ComposeTrans(transformers)

    def get_image(self, path, additional):
        '''
        Returns a transformed image by its absolute path.
        If cache is used - transformed image will be loaded if available,
        and saved to cache if not.
        '''
        if not self.use_cache:
            image = Image.open(path)
            return self.composed_all_transformers(image, additional)

        for key, post_transformers in self.split_transformers:
            if self.cache.exists(path, key):
                try:
                    image = self.cache.get(path, key)
                    image = apply_transformers_and_cache(
                        image,
                        additional,
                        path,
                        post_transformers,
                        self.cache,
                        cache_full_size=False,
                        base_key=key)
                    return image
                except:
                    hashed_key = md5(path)
                    corrupted_file = self.cache._file_path(key, hashed_key)
                    warnings.warn(CORUPTED_FILE_ERR.format(
                                                   sys.exc_info()[0]))
                    self.cache.rem(path, key)

        all_transformers = self.split_transformers[-1][1]
        image = Image.open(path)
        image = apply_transformers_and_cache(image, additional, path, all_transformers,
                                             self.cache)
        return image

    def get_images(self, paths, additionals):
        '''
        Returns a stack of transformed images by their absolute paths.
        If cache is used - transformed images will be loaded if available,
        and saved to cache if not.
        '''
        additionals += [None] * (len(paths) - len(additionals))
        images = [self.get_image(path, additional) for path, additional in zip(paths, additionals)]
        images = torch.stack(images)

        # Convert from (T, C, H, W) to (C, T, H, W)
        images = images.permute(1, 0, 2, 3)

        return images
