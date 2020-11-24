# Implements a variety of resnet models

from collections import OrderedDict
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from onconet.models.blocks.bottleneck import Bottleneck
from onconet.models.blocks.basic_block import BasicBlock
from onconet.models.factory import RegisterModel, load_pretrained_weights, get_layers
from onconet.models.resnet_base import ResNet

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def load_pretrained_model(name):
    state_dict = model_zoo.load_url(model_urls[name])
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if not 'layer' in key and not 'fc' in key:
            '''
                New name conventions places original downsampling layers
                under a downsampler module. So rename layer to downsampler.layer
            '''

            new_key = 'downsampler.{}'.format(key)
        elif 'layer' in key:
            '''
                New name conventions makes block layers under flatnamespace.
                Change layer1.block2.weight_blah to layer1_block2.weight_blah
            '''
            k_list = key.split('.')
            new_key = '{}_{}'.format(k_list[0], '.'.join(k_list[1:]))
        else:
            new_key = key
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict


@RegisterModel("resnet18")
class Default_Resnet18(nn.Module):
    def __init__(self, args):
        super(Default_Resnet18, self).__init__()
        block_layout = [[('BasicBlock', 2)],
                        [('BasicBlock', 2)],
                        [('BasicBlock', 2)],
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


@RegisterModel("resnet34")
class Default_Resnet34(nn.Module):
    def __init__(self, args):
        super(Default_Resnet34, self).__init__()
        block_layout = [[('BasicBlock', 3)],
                        [('BasicBlock', 4)],
                        [('BasicBlock', 6)],
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

@RegisterModel("resnet50")
class Default_Resnet50(nn.Module):
    def __init__(self, args):
        super(Default_Resnet50, self).__init__()
        block_layout = [[('Bottleneck', 3)],
                        [('Bottleneck', 4)],
                        [('Bottleneck', 6)],
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

@RegisterModel("resnet101")
class Default_Resnet101(nn.Module):
    def __init__(self, args):
        super(Default_Resnet101, self).__init__()
        block_layout = [[('Bottleneck', 3)],
                        [('Bottleneck', 4)],
                        [('Bottleneck', 23)],
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

@RegisterModel("resnet152")
class Default_Resnet152(nn.Module):
    def __init__(self, args):
        super(Default_Resnet152, self).__init__()
        block_layout = [[('Bottleneck', 3)],
                        [('Bottleneck', 8)],
                        [('Bottleneck', 36)],
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

@RegisterModel("deep_resnet36")
class Default_8StageResnet36(nn.Module):
    def __init__(self, args):
        super(Default_8StageResnet36, self).__init__()
        block_layout = [[('BasicBlock', 2)],
                        [('BasicBlock', 2)],
                        [('BasicBlock', 2)],
                        [('BasicBlock', 2)],
                        [('BasicBlock', 2)],
                        [('BasicBlock', 2)],
                        [('BasicBlock', 2)],
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
