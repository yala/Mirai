import torch
import torch.nn as nn
from onconet.models.pools.abstract_pool import AbstractPool
from onconet.models.pools.factory import RegisterPool
import pdb

@RegisterPool('SlidingWindowPool')
class SlidingWindow2d(AbstractPool):

    def __init__(self, args, num_chan):
        super(SlidingWindow2d, self).__init__(args, num_chan)
        ## Note, resnet downsamples image by a factor of 32 by last layer
        self.patch_maxpool = nn.MaxPool2d(args.patch_size[0] // 32)
        self.pred_fc = nn.Conv2d(args.hidden_dim, args.num_classes,
                            kernel_size=1, bias=True)
    def replaces_fc(self):
        return True

    def forward(self, x):
        patches = self.patch_maxpool(x)
        patch_preds = self.pred_fc(patches)
        B, num_class = patch_preds.size()[0], patch_preds.size()[1]
        spatially_flat_size = (B, num_class, -1)
        scores = patch_preds.view(spatially_flat_size)
        _, max_indx = torch.max(scores[:,1,:], dim=-1)
        max_indx = max_indx.unsqueeze(-1).unsqueeze(-1)
        max_indx = torch.cat([max_indx, max_indx], dim=1)
        logit = torch.gather(scores, 2, max_indx).squeeze(-1)
        return logit, patch_preds
