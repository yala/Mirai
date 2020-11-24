"""This file defines various torchvision model types in the model factory."""

import torch.nn as nn
import torchvision

from onconet.models.factory import RegisterModel

@RegisterModel("vgg11")
class Default_VGG11(nn.Module):
    def __init__(self, args):
        super(Default_VGG11, self).__init__()
        self._model = torchvision.models.vgg11(
            pretrained=args.pretrained_on_imagenet)
        args.wrap_model = True

    def forward(self, x, risk_factors=None, batch=None):
        return self._model(x)

class Default_VGG19(nn.Module):
    def __init__(self, args):
        super(Default_VGG19, self).__init__()
        self._model = torchvision.models.vgg19(
            pretrained=args.pretrained_on_imagenet)
        args.wrap_model = True

    def forward(self, x, risk_factors=None, batch=None):
        return self._model(x)



@RegisterModel("inception_v3")
class Default_InceptionV3(nn.Module):
    def __init__(self, args):
        super(Default_InceptionV3, self).__init__()
        self._model = torchvision.models.inception_v3(
            pretrained=args.pretrained_on_imagenet)
        args.wrap_model = True

    def forward(self, x, risk_factors=None, batch=None):
        return self._model(x)
