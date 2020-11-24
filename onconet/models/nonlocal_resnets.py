# Non-local Neural Networks: https://arxiv.org/abs/1711.07971

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from onconet.models.blocks.basic_block import BasicBlock
from onconet.models.blocks.bottleneck import Bottleneck
from onconet.models.factory import RegisterModel, load_pretrained_weights, get_layers
from onconet.models.default_resnets import load_pretrained_model
from onconet.models.resnet_base import ResNet

@RegisterModel("nonlocal_resnet18")
class Nonlocal_Resnet18(nn.Module):
    def __init__(self, args):
        super(Nonlocal_Resnet18, self).__init__()
        block_layout = [[('BasicBlock', 2)],
                        [('BasicBlock', 2)],
                        [('BasicBlock', 2), ('NonLocalBlock', 1)],
                        [('BasicBlock', 2)]]
        layers = get_layers(block_layout)
        self._model = ResNet(layers, args)
        if args.pretrained_on_imagenet:
              load_pretrained_weights(self._model,
                                      load_pretrained_model('resnet18'))
    def forward(self, x, risk_factors=None, batch=None):
        return self._model(x, risk_factors=risk_factors, batch=batch)

    def cuda(self, device=None):
        self._model = self._model.cuda(device)
        return self

@RegisterModel("nonlocal_resnet34")
class Nonlocal_Resnet34(nn.Module):
    def __init__(self, args):
        super(Nonlocal_Resnet34, self).__init__()
        block_layout = [[('BasicBlock', 3)],
                        [('BasicBlock', 4)],
                        [('BasicBlock', 6), ('NonLocalBlock', 1)],
                        [('BasicBlock', 3)]]
        layers = get_layers(block_layout)
        self._model = ResNet(layers, args)
        if args.pretrained_on_imagenet:
              load_pretrained_weights(self._model,
                                      load_pretrained_model('resnet34'))

    def forward(self, x, risk_factors=None, batch=None):
        return self._model(x, risk_factors=risk_factors, batch=batch)

    def cuda(self, device=None):
        self._model = self._model.cuda(device)
        return self

@RegisterModel("nonlocal_resnet50")
class Nonlocal_Resnet50(nn.Module):
    def __init__(self, args):
        super(Nonlocal_Resnet50, self).__init__()
        block_layout = [[('Bottleneck', 3)],
                        [('Bottleneck', 4)],
                        [('Bottleneck', 6), ('NonLocalBlock', 1)],
                        [('Bottleneck', 3)]]
        layers = get_layers(block_layout)
        self._model = ResNet(layers, args)
        if args.pretrained_on_imagenet:
              load_pretrained_weights(self._model,
                                    load_pretrained_model('resnet50'))

    def forward(self, x, risk_factors=None, batch=None):
        return self._model(x, risk_factors=risk_factors, batch=batch)

    def cuda(self, device=None):
        self._model = self._model.cuda(device)
        return self

@RegisterModel("nonlocal_resnet101")
class Nonlocal_Resnet101(nn.Module):
    def __init__(self, args):
        super(Nonlocal_Resnet101, self).__init__()
        block_layout = [[('Bottleneck', 3)],
                        [('Bottleneck', 4)],
                        [('Bottleneck', 23), ('NonLocalBlock', 1)],
                        [('Bottleneck', 3)]]
        layers = get_layers(block_layout)
        self._model = ResNet(layers, args)
        if args.pretrained_on_imagenet:
              load_pretrained_weights(self._model,
                                      load_pretrained_model('resnet101'))

    def forward(self, x, risk_factors=None, batch=None):
        return self._model(x, risk_factors=risk_factors, batch=batch)

    def cuda(self, device=None):
        self._model = self._model.cuda(device)
        return self

@RegisterModel("nonlocal_resnet152")
class Nonlocal_Resnet152(nn.Module):
    def __init__(self, args):
        super(Nonlocal_Resnet152, self).__init__()
        block_layout = [[('Bottleneck', 3)],
                        [('Bottleneck', 8)],
                        [('Bottleneck', 36), ('NonLocalBlock', 1)],
                        [('Bottleneck', 3)]]
        layers = get_layers(block_layout)
        self._model = ResNet(layers, args)
        if args.pretrained_on_imagenet:
              load_pretrained_weights(self._model,
                                      load_pretrained_model('resnet152'))

    def forward(self, x, risk_factors=None, batch=None):
        return self._model(x, risk_factors=risk_factors, batch=batch)

    def cuda(self, device=None):
        self._model = self._model.cuda(device)
        return self
