import torch
import torch.nn as nn
from collections import OrderedDict
import pdb

'''
    Acknowledgements, all inflation helper functions adapted from:
    Yana Hasson's github.com/hassony2/inflated_convnets_pytorch.
    I haven't submoduled the repo because I don't want all the code, and have the clone dependency. Anyways, shout out to her.
'''

INVALID_SIZE_TYPE_ERR = '{} is not expected size type'
INVALID_POOL_TYPE_ERR = '{} is not among known pooling classes'

def inflate_model(model,
                 time_dim=1,
                 time_padding=0,
                 time_stride=1,
                 time_dilation=1,
                 center=False):
    '''
        args:
        - model: nn.Module to inflate to 3d
        - time_dim: depth of time dim to inflate model to.
        - time_padding: padding of kernels in time dim
        - time_stride: stride of kernels in time dim
        - time_dilation: dilation  of kernels in time dim
        - center: initialization scheme for 3d convs. see inflate_conv

        returns:
        model (nn.Module) with all convs, pools, bns and linear layers
        inflated to 3d. For idea of model inflation, see:
        https://arxiv.org/pdf/1705.07750.pdf
    '''
    LAYER_TO_INFLATE_HELPER = {
        nn.Linear: inflate_linear,
        nn.Conv2d: inflate_conv,
        nn.MaxPool2d: inflate_pool,
        nn.AvgPool2d: inflate_pool,
        nn.BatchNorm2d: inflate_batch_norm
    }

    inflation_args = {
        'time_dim': time_dim,
        'time_padding': time_padding,
        'time_stride': time_stride,
        'time_dilation': time_dilation,
        'center':center
    }

    all_children = list(model.named_children() )
    if len(all_children) == 0:
        return model
    else:
        for name, module in all_children:

            no_layer_match = True
            for layer_type, helper in LAYER_TO_INFLATE_HELPER.items():
                if isinstance(module, layer_type):
                    module = helper(module, **inflation_args)
                    model._modules[name] = module
                    no_layer_match = False

            if no_layer_match:
                module = inflate_model(module, **inflation_args)
                model._modules[name] = module

        return model

def inflate_conv(conv2d, time_dim, time_padding,
                 time_stride, time_dilation, center):
    '''
        args:
        - conv2d: conv2d module
        - time_dim: new time dim for pool kernel. represents volume in time
        - time_padding: padding of kernel in time dim
        - time_stride: stride of pool kernel in time dim
        - time_dilation: dilation in time dim
        - center: initialization scheme for 3d conv. center = false
        means weights will be repeated across conv and normalized to preserve
        norm. center = true means center dim will contain original weights,
        rest will contain 0, shown to have slightly better performance in:
        https://arxiv.org/pdf/1712.09184.pdf

        returns:
        conv3d layer (max or avg), with all properties of original pool func
        preserved, and augmented by time args
    '''

    # To preserve activations, padding should be by continuity and not zero
    # or no padding in time dimension
    o_kernel_size = get_tuple(conv2d.kernel_size)
    o_padding = get_tuple(conv2d.padding)
    o_stride = get_tuple(conv2d.stride)
    o_dilation = get_tuple(conv2d.dilation)

    kernel_dim = (time_dim, o_kernel_size[0], o_kernel_size[1])
    padding = (time_padding, o_padding[0], o_padding[1])
    stride = (time_stride, o_stride[0], o_stride[1])
    dilation = (time_dilation, o_dilation[0], o_dilation[1])

    conv3d = nn.Conv3d(
        conv2d.in_channels,
        conv2d.out_channels,
        kernel_dim,
        padding=padding,
        dilation=dilation,
        stride=stride)
    # Repeat filter time_dim times along time dimension
    weight_2d = conv2d.weight.data
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim

    # Assign new params
    conv3d.weight = nn.Parameter(weight_3d)
    conv3d.bias = conv2d.bias
    return conv3d


def inflate_linear(linear2d, time_dim, time_padding,
                   time_stride, time_dilation, center):
    '''
        args:
        - lineard2d: linear layer in 2d setting to inflate
        - time_dim: final time dimension of the features
        - time_padding: not used in this func
        - time_stride: not used in this func
        - time_dilation: not used in this func
        - center: not used in this func

        returns:
        linear3d: linear layer with input dim scaled for num of time inputs
    '''
    linear3d = nn.Linear(linear2d.in_features * time_dim,
                               linear2d.out_features)
    weight3d = linear2d.weight.data.repeat(1, time_dim)
    weight3d = weight3d / time_dim

    linear3d.weight = nn.Parameter(weight3d)
    linear3d.bias = linear2d.bias
    return linear3d


def inflate_batch_norm(batch2d, time_dim, time_padding,
                    time_stride, time_dilation, center):
    '''
        args:
        - batch2d: BatchNorm2d
        - time_dim: not used in this func
        - time_padding: not used in this func
        - time_stride: not used in this func
        - time_dilation: not used in this func
        - center: not used in this func

        returns:
        BatchNorm3d layer , with all properties of original bn func
        preserved
    '''
    batch3d = torch.nn.BatchNorm3d(batch2d.num_features)
    batch3d.weight = batch2d.weight
    batch3d.bias = batch2d.bias
    batch3d.running_mean = batch2d.running_mean
    batch3d.running_var = batch2d.running_var
    batch3d.affine = batch2d.affine
    batch3d.momentum = batch2d.momentum
    batch3d.eps = batch2d.eps

    return batch3d


def inflate_pool(pool2d, time_dim, time_padding,
                 time_stride, time_dilation, center):
    '''
        args:
        - pool2d: maxpool2d or avgpool2d module
        - time_dim: new time dim for pool kernel. represents volume in time
        - time_padding: padding of kernel in time dim
        - time_stride: stride of pool kernel in time dim
        - time_dilation: dilation in time dim
        - center: not used in this func, maintained for consitent helper func
        args

        returns:
        3dpool layer (max or avg), with all properties of original pool func
        preserved, and augmented by time args
    '''
    o_kernel_size = get_tuple(pool2d.kernel_size)
    o_padding = get_tuple(pool2d.padding)
    o_stride = get_tuple(pool2d.stride)

    kernel_dim = (time_dim, o_kernel_size[0], o_kernel_size[1])
    padding = (time_padding, o_padding[0], o_padding[1])
    stride = (time_stride, o_stride[0], o_stride[1])

    if isinstance(pool2d, torch.nn.MaxPool2d):
        o_dilation = get_tuple(pool2d.dilation)
        dilation = (time_dilation, o_dilation[0], o_dilation[1])
        pool3d = nn.MaxPool3d( kernel_dim,
                                padding=padding,
                                dilation=dilation,
                                stride=stride,
                                ceil_mode=pool2d.ceil_mode)
    elif isinstance(pool2d, torch.nn.AvgPool2d):
        pool3d = nn.AvgPool3d( kernel_dim,
                                padding=padding,
                                stride=stride,
                                ceil_mode=pool2d.ceil_mode)
    else:
        raise ValueError(INVALID_POOL_TYPE_ERR.format(type(pool2d)))

    return pool3d

def get_tuple(size):
    '''
    args:
    - size: int or tuple representing some size, like padding or stride.
    can be int or (int, int)

    returns:
    a 2-tuple of the size, of form (int, int)
    '''
    if isinstance(size, tuple):
        return size
    elif isinstance(size, int):
        return (size, size)
    else:
        raise ValueError(INVALID_SIZE_TYPE_ERR.format(type(size)))
