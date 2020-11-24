import random
import torchvision
from onconet.transformers.factory import RegisterImageTransformer
from onconet.transformers.abstract import Abstract_transformer
import pdb


@RegisterImageTransformer("scale_3d")
class Scale_3d(Abstract_transformer):
    """Given an array of PIL images, rescales
    each image according to args.img_size.
    """

    def __init__(self, args, kwargs):
        super(Scale_3d, self).__init__()
        assert len(kwargs) == 0
        height, width = args.img_size

        def scale_3d(vid):
            return [torchvision.transforms.Resize((height, width))(img) for img in vid]

        self.transform = torchvision.transforms.Lambda(scale_3d)

    def __call__(self, vid, additional=None):
        return self.transform(vid)


@RegisterImageTransformer("random_scale_3d")
class Random_Scale_3d(Abstract_transformer):
    """Given an array of PIL images, rescale each
    so that the shorter side is the same random length
    in the range [min,max].
    """

    def __init__(self, args, kwargs):
        super(Random_Scale_3d, self).__init__()
        assert all([k in kwargs for k in ['min', 'max']])
        size = random.randint(int(kwargs['min']), int(kwargs['max']))

        def random_scale_3d(vid):
            return [torchvision.transforms.Resize(size)(img) for img in vid]

        self.transform = torchvision.transforms.Lambda(random_scale_3d)

    def __call__(self, vid, additional=None):
        return self.transform(vid)


@RegisterImageTransformer("random_crop_3d")
class Random_Crop_3d(Abstract_transformer):
    """Given an array of PIL images, randomly crop
    every image in the same location.
    """

    def __init__(self, args, kwargs):
        super(Random_Crop_3d, self).__init__()
        assert all([k in kwargs for k in ['height', 'width']])
        self.output_size = (int(kwargs['height']), int(kwargs['width']))

        def random_crop_3d(vid):
            i, j, h, w = torchvision.transforms.RandomCrop.get_params(vid[0], self.output_size)
            vid = [torchvision.transforms.functional.crop(img, i, j, h, w) for img in vid]
            return vid

        self.transform = torchvision.transforms.Lambda(random_crop_3d)

    def __call__(self, vid, additional=None):
        return self.transform(vid)


@RegisterImageTransformer("rand_hor_flip_3d")
class Random_Horizontal_Flip_3d(Abstract_transformer):
    """Randomly flips all PIL images in an array."""

    def __init__(self, args, kwargs):
        super(Random_Horizontal_Flip_3d, self).__init__()
        assert len(kwargs) == 0

        def rand_hor_flip_3d(vid):
            if random.random() < 0.5:
                vid = [torchvision.transforms.functional.hflip(img) for img in vid]
            return vid

        self.transform = torchvision.transforms.Lambda(rand_hor_flip_3d)

    def __call__(self, vid, additional=None):
        return self.transform(vid)
