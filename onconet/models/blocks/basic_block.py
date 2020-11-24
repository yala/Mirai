import torch.nn as nn
from onconet.models.blocks.factory import RegisterBlock
import pdb

def conv3x3(inplanes, outplanes, stride=1, groups=1):
    """Builds a 3x3 convolution layer with padding.

    Arguments:
        inplanes(int): The depth (number of channels)
            of the input.
        outplanes(int): The depth (number of channels)
            of the output.
        stride(int): The stride to use in the convolutions.

    Returns:
        A Conv2d layer performing 3x3 convolutions.
    """

    return nn.Conv2d(inplanes, outplanes, kernel_size=3,
                     stride=stride, padding=1, bias=False, groups=1)

@RegisterBlock('BasicBlock')
class BasicBlock(nn.Module):
    """A basic block for Resnets.

    Used in Resnet-18 and Resnet-34.
    """

    expansion = 1

    def __init__(self, args, inplanes, planes, stride=1, downsample=None):
        """Initializes the BasicBlock.

        Arguments:
            inplanes(int): The depth (number of channels) of the input.
            planes(int): The number of filters to use in convolutions
                and therefore the depth (number of channels) of the output.
            stride(int): The stride to use in the convolutions.
            downsample(func): The downsampling function to use. None if
                no downsampling is desired.
        """

        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride, groups=args.num_groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, groups=args.num_groups)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
