from torch import nn

from onconet.models.factory import RegisterModel, load_pretrained_weights, get_layers
from onconet.models.default_resnets import load_pretrained_model
from onconet.models.resnet_base import ResNet

@RegisterModel("custom_resnet")
class CustomResnet(nn.Module):
    def __init__(self, args):
        super(CustomResnet, self).__init__()
        layers = get_layers(args.block_layout)
        self._model = ResNet(layers, args)
        model_name = args.pretrained_imagenet_model_name
        if args.pretrained_on_imagenet:
            load_pretrained_weights(self._model,
                                    load_pretrained_model(model_name))

    def forward(self, x, risk_factors=None, batch=None):
        return self._model(x, risk_factors=risk_factors, batch=None)

    def cuda(self, device=None):
        self._model = self._model.cuda(device)
        return self
