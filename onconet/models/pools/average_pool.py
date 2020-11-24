import torch
import torch.nn as nn
from onconet.models.pools.abstract_pool import AbstractPool
from onconet.models.pools.factory import RegisterPool


@RegisterPool('GlobalAvgPool')
class GlobalAvgPool(AbstractPool):

    def replaces_fc(self):
        return False

    def forward(self, x):
        spatially_flat_size = (*x.size()[:2], -1)
        x = x.view(spatially_flat_size)
        x = torch.mean(x, dim = -1)

        return None, x
