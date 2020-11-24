import torch.nn as nn
from onconet.models.blocks.factory import RegisterBlock
from onconet.models.blocks.basic_block import conv3x3
import pdb


@RegisterBlock('PreactBasic')
class PreactBasicBlock(nn.Module):
    """A wide basic block for Resnets, from "Wide Residual Networks"

    https://arxiv.org/pdf/1605.07146.pdf
    """

    expansion = 1

    def __init__(self, args, inplanes, planes, stride=1, downsample=None):
        """Initializes

        Arguments:
            inplanes(int): The depth (number of channels) of the input.
            planes(int): The number of filters to use in convolutions
                and therefore the depth (number of channels) of the output.
            stride(int): The stride to use in the convolutions.
            downsample(func): The downsampling function to use. None if
                no downsampling is desired.
        """

        super(PreactBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride, groups=args.num_groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, groups=args.num_groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(p=args.dropout)


    def forward(self, x):
        """Computes a forward pass of the model.

        Arguments:
            x(Variable): The input to the model.

        Returns:
            The result of feeding the input through the model.
        """

        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

