import torch
import torch.nn as nn
import torch.nn.functional as F
from onconet.models.spatial_transformers.factory import RegisterSpatialTransformer
from onconet.models.factory import get_model
import pdb
import copy

NUM_PARAMS_FOR_AFFINE_TRANSFORM = 6
MULTI_IMG_LOC_NET_POOL_NAME = "LinearConcat_MaxPool"
SINGLE_IMG_LOC_NET_POOL_NAME = "GlobalMaxPool"

@RegisterSpatialTransformer('affine')
class AffineSpatialTransformer(nn.Module):

    def __init__(self, args):
        super(AffineSpatialTransformer, self).__init__()
        self.args = args
        loc_net_args = copy.deepcopy(args)
        loc_net_args.model_name = args.location_network_name
        loc_net_args.block_layout = args.location_network_block_layout
        loc_net_args.pool_name = MULTI_IMG_LOC_NET_POOL_NAME if args.multi_image else SINGLE_IMG_LOC_NET_POOL_NAME
        loc_net_args.use_spatial_transformer = False
        self.loc_net = get_model(loc_net_args)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(args.dropout)
        self.fc_loc = nn.Linear(loc_net_args.hidden_dim // args.num_images, NUM_PARAMS_FOR_AFFINE_TRANSFORM)
        self.fc_loc.weight.data.zero_()
        self.fc_loc.bias.data.copy_(torch.FloatTensor([1, 0, 0, 0, 1, 0]))

    def localize(self, x):
        # TODO, figureout how to have really low res X for this stage
        _, loc_hidden, _ = self.loc_net(x)
        if self.args.multi_image:
            B, _, T, _, _ = x.size()
            loc_hidden = loc_hidden.view([B, loc_hidden.size()[-1] // T, T])
            loc_hidden = loc_hidden.permute([0,2,1]).contiguous()
        loc_hidden = self.dropout(self.relu(loc_hidden))
        theta = self.fc_loc(loc_hidden)
        theta = theta.view(-1,2,3)
        return theta

    def grid_sample(self, x, theta):
        if self.args.multi_image:
            B, C, T, H, W = x.size()
            x = x.permute([0,2,1,3,4])
            x = x.contiguous().view(-1, C, H, W)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        if self.args.multi_image:
            x = x.view([B, T, C, H, W]).permute([0,2,1,3,4]).contiguous()
        return x

    def forward(self, x):
        theta = self.localize(x)
        tranf_x = self.grid_sample(x, theta)
        return tranf_x


