import torch
import torch.nn as nn
from onconet.models.pools.abstract_pool import AbstractPool
from onconet.models.pools.factory import RegisterPool
from onconet.models.pools.factory import get_pool
import torch.autograd as autograd

import pdb

@RegisterPool('LinearConcat_MaxPool')
class LinearConcat_MaxPool(AbstractPool):
    def __init__(self, args, num_chan):
        super(LinearConcat_MaxPool, self).__init__(args, num_chan)
        assert args.multi_image
        self.args = args

        self.args.hidden_dim = num_chan * self.args.num_images

    def replaces_fc(self):
        return False

    def forward(self, x):
        # reshape x, from shape (B, C, T, W, H) to (B, CT)

        # Get shape (B, C, T, -1)
        spatially_flat_size = (*x.size()[:3], -1)
        x = x.view(spatially_flat_size)
        x, _ = torch.max(x, dim=-1)
        x = x.view( (spatially_flat_size[0],-1))
        return None, x


@RegisterPool('LinearConcat_MaxPool_SubDot')
class LinearConcat_MaxPool_SubDot(AbstractPool):
    def __init__(self, args, num_chan):
        super(LinearConcat_MaxPool_SubDot, self).__init__(args, num_chan)
        assert args.multi_image
        assert args.num_images == 2
        self.args = args

        self.args.hidden_dim = num_chan * self.args.num_images

    def replaces_fc(self):
        return False

    def forward(self, x):
        # reshape x, from shape (B, C, T, W, H) to (B, CT)

        # Get shape (B, C, T, -1)
        spatially_flat_size = (*x.size()[:3], -1)
        x = x.view(spatially_flat_size)
        x, _ = torch.max(x, dim=-1)
        view1, view2 = x[:,:,0], x[:,:,1]
        sub, dot = torch.abs(view1 - view2), (view1 * view2)
        x = torch.cat([sub, dot], dim=1)
        return None, x



@RegisterPool('LinearConcat_MaxPool_ToDense')
class LinearConcat_MaxPool_ToDense(AbstractPool):
    def __init__(self, args, num_chan):
        super(LinearConcat_MaxPool_ToDense, self).__init__(args, num_chan)
        assert args.multi_image
        assert args.num_images == 2
        self.args = args
        self.bn = nn.BatchNorm1d(args.num_images * args.hidden_dim)
        self.fc = nn.Linear( args.num_images * args.hidden_dim, args.hidden_dim)
        self.relu = nn.ReLU(inplace=True)


    def replaces_fc(self):
        return False

    def forward(self, x):
        # reshape x, from shape (B, C, T, W, H) to (B, CT)

        # Get shape (B, C, T, -1)
        spatially_flat_size = (*x.size()[:3], -1)
        x = x.view(spatially_flat_size)
        x, _ = torch.max(x, dim=-1)
        x = x.view( (spatially_flat_size[0],-1))
        x = self.fc(self.relu(self.bn(x)))
        return None, x


@RegisterPool('BiLinearConcat_MaxPool')
class BiLinearConcat_MaxPool(AbstractPool):
    def __init__(self, args, num_chan):
        super(BiLinearConcat_MaxPool, self).__init__(args, num_chan)
        assert args.multi_image
        assert args.num_images == 2
        self.args = args
        self.dropout = nn.Dropout(p=args.dropout)
        self.bilinear = nn.Bilinear(self.args.hidden_dim, self.args.hidden_dim, args.num_classes)


    def replaces_fc(self):
        return True

    def forward(self, x):
        # reshape x, from shape (B, C, T, W, H) to (B, CT)

        # Get shape (B, C, T, -1)
        spatially_flat_size = (*x.size()[:3], -1)
        x = x.view(spatially_flat_size)
        # Reduce to shape (B,C,T)
        x, _ = torch.max(x, dim=-1)
        x = self.dropout(x)

        hidden = x.view( (spatially_flat_size[0],-1))
        logit = self.bilinear(x[:,:,0], x[:,:,1])
        return logit, hidden
