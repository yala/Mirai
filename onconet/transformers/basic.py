import torch
import torchvision
from onconet.transformers.abstract import Abstract_transformer


class ToTensor(Abstract_transformer):
    '''
    torchvision.transforms.ToTensor wrapper.
    '''

    def __init__(self):
        super(ToTensor, self).__init__()
        self.transform = torchvision.transforms.ToTensor()

    def __call__(self, img, additional=None):
        return self.transform(img).float()


class ToTensor3d(Abstract_transformer):
    """Convert a length T list of PIL images (H, W, C)
    to a 4D PyTorch tensor (T, C, H, W)"""

    def __init__(self):
        super(ToTensor3d, self).__init__()

        def to_tensor_3d(vid):
            return torch.stack([torchvision.transforms.ToTensor()(img) for img in vid])

        self.transform = torchvision.transforms.Lambda(to_tensor_3d)

    def __call__(self, vid, additional=None):
        return self.transform(vid).float()


class ToPIL3d(Abstract_transformer):
    """Convert a 4D array (T, H, W, C)
    to an array of PIL images (H, W, C).
    """

    def __init__(self):
        super(ToPIL3d, self).__init__()

        def to_pil_3d(vid):
            return [torchvision.transforms.ToPILImage()(img) for img in vid]

        self.transform = torchvision.transforms.Lambda(to_pil_3d)

    def __call__(self, vid, additional=None):
        return self.transform(vid)


class Permute3d(Abstract_transformer):
    """Permute tensor (T, C, H, W) ==> (C, T, H, W)"""

    def __init__(self):
        super(Permute3d, self).__init__()

        def permute_3d(tensor):
            return tensor.permute(1, 0, 2, 3)

        self.transform = torchvision.transforms.Lambda(permute_3d)

    def __call__(self, tensor, additional=None):
        return self.transform(tensor)


class ComposeTrans(Abstract_transformer):
    '''
    composes multiple transformers
    '''

    def __init__(self, transformers):
        super(ComposeTrans, self).__init__()
        self.transformers = transformers

    def __call__(self, img, additional=None):
        for transformer in self.transformers:
            img = transformer(img, additional)

        return img
