import torchvision
import torch
from onconet.transformers.factory import RegisterTensorTransformer
import numpy as np
from onconet.transformers.abstract import Abstract_transformer
import pickle

@RegisterTensorTransformer("normalize_2d")
class Normalize_Tensor_2d(Abstract_transformer):
    '''
    torchvision.transforms.Normalize wrapper.
    '''

    def __init__(self, args, kwargs):
        super(Normalize_Tensor_2d, self).__init__()
        assert len(kwargs) == 0
        channel_means = [args.img_mean] if len(args.img_mean) == 1 else args.img_mean
        channel_stds = [args.img_std] if len(args.img_std) == 1 else args.img_std

        self.transform = torchvision.transforms.Normalize(torch.Tensor(channel_means),
                                                          torch.Tensor(channel_stds))

    def __call__(self, img, additional=None):
        return self.transform(img)


@RegisterTensorTransformer("cutout")
class CutOut(Abstract_transformer):
    '''
        Randomly sets a patch to black.
        size of patch will be decided by the 'h' and 'w' kwargs. Done with probablity p
        From: https://arxiv.org/pdf/1708.04552.pdf
    '''

    def __init__(self, args, kwargs):
        super(CutOut, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len == 3
        mask_w, mask_h, p = (int(kwargs['w']), int(kwargs['h']), float(kwargs['p']))
        img_w, img_h = self.args.img_size
        mask = 0

        def cutout(image):
            if np.random.random() > p:
                return image
            center_x, center_y = np.random.randint(0, img_w), np.random.randint(0, img_h)

            x_min, x_max = center_x - (mask_w // 2), center_x + (mask_w // 2)
            y_min, y_max = center_y - (mask_h // 2), center_y + (mask_h // 2)
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(img_w, y_min), min(img_h, y_max)
            image[y_min:y_max, x_min:x_max] *= mask

            return image
        self.transform = torchvision.transforms.Lambda(cutout)

    def __call__(self, img, additional=None):
        return self.transform(img)


@RegisterTensorTransformer("normalize_3d")
class Normalize_Tensor_3d(Abstract_transformer):
    '''
    torchvision.transforms.Normalize wrapper.
    '''

    def __init__(self, args, kwargs):
        super(Normalize_Tensor_3d, self).__init__()
        assert len(kwargs) == 0
        channel_means = [args.img_mean] * args.num_chan if len(args.img_mean) == 1 else args.img_mean
        channel_stds = [args.img_std] * args.num_chan if len(args.img_std) == 1 else args.img_std

        def normalized_tensor_3d(tensor):
            norm = torchvision.transforms.Normalize(torch.Tensor(channel_means),
                                                    torch.Tensor(channel_stds))
            return torch.stack([norm(img_tensor) for img_tensor in tensor])

        self.transform = torchvision.transforms.Lambda(normalized_tensor_3d)

    def __call__(self, img, additional=None):
        return self.transform(img)


@RegisterTensorTransformer("channel_shift")
class Channel_Shift_Tensor(Abstract_transformer):
    '''
    Randomly shifts values in a channel by a random number uniformly sampled
    from -shift:shift.
    '''

    def __init__(self, args, kwargs):
        super(Channel_Shift_Tensor, self).__init__()
        assert len(kwargs) == 1
        shift = float(kwargs['shift'])

        def apply_shift(img):
            shift_val = float(np.random.uniform(low=-shift, high=shift, size=1))
            return img + shift_val

        self.transform = torchvision.transforms.Lambda(apply_shift)

    def __call__(self, img, additional=None):
        return self.transform(img)


@RegisterTensorTransformer("force_num_chan_2d")
class Force_Num_Chan_Tensor_2d(Abstract_transformer):
    '''
    Convert gray scale images to image with args.num_chan num channels.
    '''

    def __init__(self, args, kwargs):
        super(Force_Num_Chan_Tensor_2d, self).__init__()
        assert len(kwargs) == 0

        def force_num_chan(tensor):
            existing_chan = tensor.size()[0]
            if not existing_chan == args.num_chan:
                return tensor.expand(args.num_chan, *tensor.size()[1:])
            return tensor

        self.transform = torchvision.transforms.Lambda(force_num_chan)

    def __call__(self, img, additional=None):
        return self.transform(img)


@RegisterTensorTransformer("force_num_chan_3d")
class Force_Num_Chan_Tensor_3d(Abstract_transformer):
    '''
    Convert a video with gray scale images to image with args.num_chan num channels.
    '''

    def __init__(self, args, kwargs):
        super(Force_Num_Chan_Tensor_3d, self).__init__()
        assert len(kwargs) == 0

        def force_num_chan(tensor):
            existing_chan = tensor.size()[1]
            if not existing_chan == args.num_chan:
                return tensor.expand(tensor.size()[0], args.num_chan, *tensor.size()[2:])
            return tensor

        self.transform = torchvision.transforms.Lambda(force_num_chan)

    def __call__(self, img, additional=None):
        return self.transform(img)
