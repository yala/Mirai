import torch
import torch.nn as nn
import pdb



'''
    Implements group normalization, https://arxiv.org/pdf/1803.08494.pdf

    Idea similar to batch norm but has show promising results in low batch scenario.
    Assumes features (channels) are at dim1 of tensor, and batch dim is dim0
'''
class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        dims = x.size()
        B, C = dims[0], dims[1]
        G = self.num_groups
        assert C % G == 0

        x = x.view(B,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(dims)
        return x * self.weight + self.bias

    def __repr__(self):
        return '{num_features}, G={num_groups}, eps={eps}'.format(**self.__dict__)
