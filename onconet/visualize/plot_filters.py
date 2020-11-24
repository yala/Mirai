# Adapted from https://discuss.pytorch.org/t/understanding-deep-network-visualize-weights/2060/8
import argparse
import math

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
import torch
from torchvision import models  # for testing
from tqdm import trange


def float_to_pixel(x):
    """Converts a tensor from float values to valid pixels values in range [0, 255]."""

    # Normalize tensor; center on 0; ensure std is 0.1
    x -= x.mean()
    x /= x.std()
    x *= 0.1

    # Clip to [0,1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # Convert to pixel values, i.e. ints in range [0, 255]
    x *= 255
    x = x.astype('uint8')

    return x


def plot_filters_bw(save_path, tensor, title='Filters'):
    """Plots all the channels of all the filters for a tensor in a grid."""

    if not tensor.ndim == 4:
        raise Exception("Assumes a 4D tensor")

    num_filters = tensor.shape[0]
    num_outer_rows = num_outer_cols = int(math.ceil(math.sqrt(num_filters)))
    num_channels = tensor.shape[1]
    num_inner_rows = num_inner_cols = int(math.ceil(math.sqrt(num_channels)))

    fig = plt.figure()
    fig.suptitle(title)
    outer = gridspec.GridSpec(
        num_outer_rows, num_outer_cols, wspace=0.4, hspace=0.4)

    for f in trange(num_filters):
        inner = gridspec.GridSpecFromSubplotSpec(
            num_inner_rows,
            num_inner_cols,
            subplot_spec=outer[f],
            wspace=0.1,
            hspace=0.1)

        for c in trange(num_channels):
            ax = plt.Subplot(fig, inner[c])
            ax.imshow(
                float_to_pixel(tensor[f, c]),
                interpolation='none',
                cmap='gray')
            ax.axis('off')
            fig.add_subplot(ax)

    plt.savefig(save_path)


def plot_filters_rgb(save_path, tensor, title='Filters'):
    """Plots RGB (3-channel) filters for a tensor in a grid."""

    if not tensor.ndim == 4:
        raise Exception("Assumes a 4D tensor")

    tensor = np.transpose(tensor, (0, 2, 3, 1))  # move rgb dim to last dim

    if not tensor.shape[-1] == 3:
        raise Exception("Last dimension needs to be 3 to plot (for RGB)")

    num_filters = tensor.shape[0]
    num_rows = num_cols = int(math.ceil(math.sqrt(num_filters)))

    fig = plt.figure(figsize=(num_rows, num_cols))
    fig.suptitle(title)

    for i in trange(num_filters):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.imshow(float_to_pixel(tensor[i]), interpolation='none')
        ax.axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(save_path)


def plot_filters(save_path, snapshot_path, layer_name, rgb):
    try:
        model = torch.load(snapshot_path)
    except:
        # model = models.resnet18(pretrained=True) # for testing
        raise Exception(
            "Sorry, snapshot {} does not exist!".format(snapshot_path))

    layer_dict = {name: param for name, param in model.named_parameters()}
    layer = layer_dict[layer_name]
    tensor = layer.data.numpy()

    if rgb:
        plot_filters_rgb(save_path, tensor, title=layer_name)
    else:
        plot_filters_bw(save_path, tensor, title=layer_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_path',
        type=str,
        required=True,
        help='Path where the plot will be saved')
    parser.add_argument(
        '--snapshot_path',
        type=str,
        required=True,
        help='Path to a model snapshot')
    parser.add_argument(
        '--layer_name',
        type=str,
        required=True,
        help='The name of the layer with filters to plot, ex. "conv1.weight"')
    parser.add_argument(
        '--rgb',
        action='store_true',
        default=False,
        help='True to plot 3-channel filters in color (RGB)')
    args = parser.parse_args()

    plot_filters(args.save_path, args.snapshot_path, args.layer_name, args.rgb)
