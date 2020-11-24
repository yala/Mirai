import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onconet.models.group_norm import GroupNorm
from onconet.models.blocks.factory import RegisterBlock
import pdb

UNEXPECTED_INPUT_SIZE_ERR = "Unexpected input size! Expected a 5D tensor, instead got size {}"

@RegisterBlock('ACABlock')
class AttendCompareAggregateBlock(nn.Module):
    """Attend Compare Aggregate block."""
    expansion = 1

    def __init__(self, args, inplanes, planes, stride=1, downsample=None,
                 compression_factor=2):
        """Initializes the ACABlock.

        Arguments:
            inplanes(int): The depth (number of channels) of the input.
            planes(int): The depth (number of channels) of the output.
            downsample(nn.Module): When not none, used to downsample output to planes.

            compression_factor(int): The compression factor to use when
                performing max pooling over attention mem and attention val to speed up computation. Note, Not implemented.
        """

        super(AttendCompareAggregateBlock, self).__init__()
        self.compression_factor = compression_factor
        self.num_images = args.num_images
        self.attend_module = AttendModule(args, inplanes, compression_factor)
        self.compare_module = CompareModule(args, inplanes, self.num_images)
        self.aggregate_module = AggregateModule(args, inplanes, inplanes, self.num_images)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        """Computes a forward pass of the model.

        Arguments:
            x(Variable): The input to the model.

        Returns:
            The result of feeding the input through the model.
        """

        # Save residual
        if self.compression_factor > 1:
            x = F.max_pool3d(x, kernel_size=(1,3,3), stride=(1,self.compression_factor,self.compression_factor))

        residual = x

        R, V = self.attend_module(x)
        C = self.compare_module(R, V)
        Z = self.aggregate_module(R, V, C)

        out = x+Z
        out = self.relu(out)

        if self.downsample is not None:
            out = self.downsample(out)

        return out


class AttendModule(nn.Module):
    """
        Attend step of ACA. For each pixel each frame, get a attention distribution over all other frames. i.e starting from a T, H, W volume,
        produce an attention map of of T, T, H, W.

    """

    def __init__(self, args, inplanes, compression_factor=2):


        super(AttendModule, self).__init__()

        self.query_conv = nn.Conv3d(inplanes,
                                    inplanes // 2,
                                    kernel_size=1,
                                    bias=False)
        self.memory_conv = nn.Conv3d(inplanes,
                                  inplanes // 2,
                                  kernel_size=1,
                                  bias=False)
        self.value_conv = nn.Conv3d(inplanes,
                                inplanes // 2,
                                kernel_size=1,
                                bias=False)


    def forward(self, x):
        """Computes a attention step of ACA block.

        Arguments:
            x(Variable): The input to the model.

        Returns:
            R -  (B, C/2, T, T, H, W) tensor representing the attended vector at each timestep for each pixel
            V - (B, C/2, T, H, W) tensor. R = A * V
        """

        dims = x.size()


        if len(dims) == 5:
            batch_size, inplanes, time, height, width = dims
        else:
            raise Exception(UNEXPECTED_INPUT_SIZE_ERR.format(dims))


        # Compute Q, M, V all of size (B, C/2, T, H, W)
        Q = self.query_conv(x)
        M = self.memory_conv(x)
        V = self.value_conv(x)

        V_i = V # Save copy of V in shape (B, C/2, T, H, W)

        # Reshape Q, M to (B, C/2, THW)
        Q = Q.view(batch_size, inplanes//2, -1)
        M = M.view(batch_size, inplanes//2, -1)
        # Reshape Q to (B, TWH, C/2)
        Q = torch.transpose(Q, 1,2)

        # Reshape V to (BT, HW, C/2)
        V = torch.transpose(V, 1, 2) #( B, T, C/2, H, W)
        V = V.contiguous().view(batch_size * time, inplanes//2, height * width)
        V = torch.transpose(V, 1, 2)

        # Compute attention, shape is (B, THW, THW)
        # Scaling by sqrt(dim) is inspired by https://arxiv.org/pdf/1706.03762.pdf
        A = torch.bmm(Q, M) / np.sqrt(inplanes // 2)
        # Reshape A (B, THW, THW) to (BT, THW, HW)
        A = A.contiguous().view(batch_size, -1, time, height*width) #( B, TWH, T, HW)
        A = torch.transpose(A, 1, 2) #( B, T, TWH, HW)
        A = A.contiguous().view( batch_size * time, -1, height*width)
        A = F.softmax( A, dim=-1)

        # Compute R = A * V, output of shape (BT, THW, C/2)
        R = torch.bmm(A, V)

        # Reshape R to (B, C/2, T, T, H, W)
        R = R.view( batch_size, time, time, height, width, inplanes //2)
        R = R.permute(0, 5, 1, 2, 3, 4)
        return R, V_i

class CompareModule(nn.Module):
    """
        Compare step of ACA. Given a feature vec Vi, compare it in the
        style of NLI to all corresponding attended vectors in R.
    """
    def __init__(self, args, inplanes, num_images):


        super(CompareModule, self).__init__()

        self.cat_conv = nn.Conv3d(inplanes,
                                inplanes // (num_images * 3),
                                kernel_size=1,
                                bias=False)

        self.sub_conv = nn.Conv3d(inplanes // 2,
                                inplanes // (num_images * 3),
                                kernel_size=1,
                                bias=False)

        self.mul_conv = nn.Conv3d(inplanes // 2,
                                inplanes // (num_images * 3),
                                kernel_size=1,
                                bias=False)

        Norm = GroupNorm if args.replace_bn_with_gn else nn.BatchNorm2d
        self.cat_bn = Norm(inplanes // (num_images * 3))
        self.sub_bn = Norm(inplanes // (num_images * 3))
        self.mul_bn = Norm(inplanes // (num_images * 3))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, R, V):
        '''
            For each pixel in V, compare to all attended time steps in R.
            via fc(V-R), fc(V*R), fc(V;R)

            args:
            R: (B, C/2, T, T, H, W) tensor
            V:  (B, C/2, T, H, W) tensor

            returns:
            C: (B, C', T, H, W, comparison features
        '''
        dims = V.size()

        if len(dims) == 5:
            batch_size, chan, time, height, width = dims
        else:
            raise Exception(UNEXPECTED_INPUT_SIZE_ERR.format(dims))

        V_exp = V.unsqueeze(0)
        V_exp = V_exp.expand(time, batch_size, chan, time, height, width)
        V_exp = V_exp.permute(1,2,3,0,4,5)
        V_exp = V_exp.contiguous().view(batch_size, chan, time*time, height, width)

        R = R.contiguous().view(batch_size, chan, time*time, height, width)

        X_cat = torch.cat([R, V_exp], dim=1)
        X_sub = R - V_exp
        X_mul = R * V_exp

        C_cat = self.cat_bn(self.cat_conv(X_cat))
        C_sub = self.sub_bn(self.sub_conv(X_sub))
        C_mul = self.mul_bn(self.mul_conv(X_mul))
        C = torch.cat([C_cat, C_sub, C_mul], dim=1)

        C = C.view(batch_size, -1, time, height, width)

        # channel dim = (inplanes // (num_images * 3)) * 3 * num_images
        C = self.relu(C)
        return C


class AggregateModule(nn.Module):
    """
        Aggregate step of ACA. Given input features, attended features,
        and comparison features, compress down to a single space and model
        cross channel dependecies.
    """
    def __init__(self, args, inplanes, planes, num_images):


        super(AggregateModule, self).__init__()
        c_dim = (inplanes // (num_images*3))*3*num_images
        in_dim = inplanes  + c_dim

        self.agg_conv = nn.Conv3d(in_dim,
                                planes,
                                kernel_size=1,
                                bias=False)

        Norm = GroupNorm if args.replace_bn_with_gn else nn.BatchNorm2d
        self.agg_bn = Norm(planes)


    def forward(self, R, V, C):
        '''
            Args:
            R: (B, C/2, T, T, W, H),  attention features
            V: Shape (B, C/2, T, W, H), input features
            C: Shape (B, C', T, W, H), comparison features

            returns,
            Z: Shape (B, C, T, W, H), aggregated result
        '''
        R = torch.mean(R, dim=3)

        Y = torch.cat([R,V,C], dim=1)

        Z = self.agg_bn(self.agg_conv(Y))
        return Z









