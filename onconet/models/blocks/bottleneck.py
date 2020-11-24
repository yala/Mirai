import torch.nn as nn
from onconet.models.blocks.factory import RegisterBlock

@RegisterBlock('Bottleneck')
class Bottleneck(nn.Module):
    """A bottleneck block for Resnets.

    Used in Resnet-50, Resnet-101, and Resnet-152.
    """

    expansion = 4

    def __init__(self, args, inplanes, planes, stride=1, downsample=None):
        """Initializes the Bottleneck.

        Arguments:
            inplanes(int): The depth (number of channels) of the input.
            planes(int): The number of filters to use in convolutions
                and therefore the depth (number of channels) of the output.
            stride(int): The stride to use in the convolutions.
            downsample(func): The downsampling function to use. None if
                no downsampling is desired.
        """

        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, groups=args.num_groups)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=args.num_groups)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, groups=args.num_groups)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Computes a forward pass of the model.

        Arguments:
            x(Variable): The input to the model.

        Returns:
            The result of feeding the input through the model.
        """

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
