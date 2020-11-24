import copy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import onconet.datasets.factory as dataset_factory
from onconet.learn.utils import ignore_None_collate
import onconet.transformers.factory as transformer_factory
from onconet.utils.parsing import parse_transformers
from onconet.utils.generic import normalize_dictionary

import sys
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))

def modify_args(args):
    # Set transformer to resize image img_size before computing stats
    # to improve computation speed and memory overhead
    args.num_chan = 3
    args.cache_path = None
    args.img_size = (256, 256)

    dim = '3d' if args.video else '2d'
    args.image_transformers = parse_transformers(['scale_{}'.format(dim)])
    args.tensor_transformers = parse_transformers(['force_num_chan_{}'.format(dim)])

def get_dataset_stats(args):
    args = copy.deepcopy(args)
    modify_args(args)

    transformers = transformer_factory.get_transformers(args.image_transformers, args.tensor_transformers, args)

    train, _, _ = dataset_factory.get_dataset(args, transformers, [])

    data_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=ignore_None_collate)

    means, stds = {0: [], 1: [], 2: []}, {0: [], 1: [], 2: []}
    indx = 1
    for batch in tqdm(data_loader):
        tensor = batch['x']

        if args.cuda:
            tensor = tensor.cuda()
        for channel in range(3):
            tensor_chan = tensor[:, channel]
            means[channel].append(torch.mean(tensor_chan))
            stds[channel].append(torch.std(tensor_chan))
        
        if indx % (len(data_loader)//20) == 0:
            _means = [torch.mean(torch.Tensor(means[channel])) for channel in range(3)]
            _stds = [torch.mean(torch.Tensor(stds[channel])) for channel in range(3)]
            print('for indx={}\t mean={}\t std={}\t'.format(indx, _means, _stds))
        indx += 1
    means = [torch.mean(torch.Tensor(means[channel])) for channel in range(3)]
    stds = [torch.mean(torch.Tensor(stds[channel])) for channel in range(3)]

    return means, stds
