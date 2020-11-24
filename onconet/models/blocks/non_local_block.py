# Implementation inspired by Keras implementation: https://github.com/titu1994/keras-non-local-nets/blob/master/non_local.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from onconet.models.group_norm import GroupNorm
from onconet.models.blocks.factory import RegisterBlock

UNEXPECTED_INPUT_SIZE_ERR = "Unexpected input size! Expected a 4D or 5D tensor, instead got size {}"

@RegisterBlock('NonLocalBlock')
class NonLocalBlock(nn.Module):
    """An embedded gaussian non-local block."""
    expansion = 1

    def __init__(self, args, inplanes, planes, stride=1, downsample=None,
                 compression_factor=2):
        """Initializes the NonLocalBlock.

        Arguments:
            inplanes(int): The depth (number of channels) of the input.
            planes(int): The depth (number of channels) of the output.
            downsample(nn.Module): When not none, used to downsample output to planes.

            compression_factor(int): The compression factor to use when
                performing max pooling over phi and g to speed up computation.
        """

        super(NonLocalBlock, self).__init__()
        self.compression_factor = 1 if args.use_precomputed_hiddens else compression_factor
        self.theta_conv = nn.Conv2d(inplanes,
                                    inplanes // 2,
                                    kernel_size=1,
                                    bias=False)
        self.phi_conv = nn.Conv2d(inplanes,
                                  inplanes // 2,
                                  kernel_size=1,
                                  bias=False)
        self.g_conv = nn.Conv2d(inplanes,
                                inplanes // 2,
                                kernel_size=1,
                                bias=False)
        self.y_conv = nn.Conv2d(inplanes // 2,
                                inplanes,
                                kernel_size=1,
                                bias=False)

        Norm = GroupNorm if args.replace_bn_with_gn else nn.BatchNorm2d
        self.y_conv_bn = Norm(inplanes)

        # Set batch norm gamma to 0
        self.y_conv_bn.weight = nn.Parameter(torch.zeros(self.y_conv_bn.weight.size()))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        """Computes a forward pass of the model.

        Arguments:
            x(Variable): The input to the model.

        Returns:
            The result of feeding the input through the model.
        """

        if self.compression_factor > 1:
            x = F.max_pool3d(x, kernel_size=(1,3,3), stride=(1,self.compression_factor,self.compression_factor))
        # Save residual
        residual = x

        # Get dimensions
        dims = x.size()
        if len(dims) == 4:
            batch_size, inplanes, height, width = dims
            size = (height, width)
        elif len(dims) == 5:
            batch_size, inplanes, time, height, width = dims
            size = (time, height, width)
        else:
            raise Exception(UNEXPECTED_INPUT_SIZE_ERR.format(dims))

        # Compute theta, phi, g all of size (B, C/2, ?T, H, W)
        theta = self.theta_conv(x)
        phi = self.phi_conv(x)
        g = self.g_conv(x)

        # Reshape theta, phi, g to size (B, C/2, (?T)HW)
        theta = theta.view(batch_size, inplanes // 2, -1)
        phi = phi.view(batch_size, inplanes // 2, -1)
        g = g.view(batch_size, inplanes // 2, -1)

        # Transpose theta and g to be (B, (?T)HW, C/2)
        theta = torch.transpose(theta, dim0=1, dim1=2)
        g = torch.transpose(g, dim0=1, dim1=2)

        # Compute f = softmax(theta * phi)
        f = F.softmax(torch.bmm(theta, phi), dim=-1)

        # Compute y = f * g
        y = torch.bmm(f, g)

        # Reshape to (B, C/2, ?T, H, W)
        out = y.view(batch_size, inplanes // 2, *size)

        # Project to (B, C, ?T, H, W)
        out = self.y_conv(out)

        # Batch norm
        out = self.y_conv_bn(out)

        # Add residual
        out += residual
        out = self.relu(out)

        if self.downsample is not None:
            out = self.downsample(out)

        return out
