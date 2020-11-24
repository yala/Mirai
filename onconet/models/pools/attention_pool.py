import torch
import torch.nn as nn
from torch.nn.functional import softmax
from onconet.models.pools.abstract_pool import AbstractPool
from onconet.models.pools.factory import RegisterPool
import numpy as np
import pdb


@RegisterPool('Simple_AttentionPool')
class Simple_AttentionPool(AbstractPool):
    def __init__(self, args, num_chan):
        super(Simple_AttentionPool, self).__init__(args, num_chan)

        self.attention_fc = nn.Linear(num_chan, 1)
        self.softmax = nn.Softmax(dim=-1)

    def replaces_fc(self):
        return False

    def forward(self, x):
        #X dim: B, C, W, H
        spatially_flat_size = (*x.size()[:2], -1) #B, C, N
        x = x.view(spatially_flat_size)
        attention_scores = self.attention_fc(x.transpose(1,2)) #B, N, 1
        attention_scores = self.softmax( attention_scores.transpose(1,2)) #B, 1, N
        x = x * attention_scores #B, C, N
        x = torch.sum(x, dim=-1)
        return None, x


@RegisterPool('AttentionPool2d')
class AttentionPool2d(AbstractPool):
    def __init__(self, args, num_chan):
        super(AttentionPool2d, self).__init__(args, num_chan)

        self.K = args.num_classes
        self.C = num_chan

        self.class_key = nn.Parameter(torch.ones(self.C, self.K), requires_grad=True)
        self.class_value = nn.Parameter(torch.ones(self.C, self.K), requires_grad=True)
        stdv = 1. / np.sqrt(self.C)
        self.class_key.data.uniform_(-stdv, stdv)
        self.class_value.data.uniform_(-stdv, stdv)
        self.register_parameter('class_key', self.class_key)
        self.register_parameter('class_val', self.class_value)

        self.activation_key_conv = nn.Conv2d(self.C, self.C, kernel_size=1, bias=False)
        self.activation_value_conv = nn.Conv2d(self.C, self.C, kernel_size=1, bias=False)

    def replaces_fc(self):
        return True

    def compute_attention(self, x):
        B = x.size()[0]
        num_dim = len(x.size())
        class_key = self.class_key.expand(B, -1, -1)  # (B, C, K)
        activation_key = self.activation_key_conv(x)  # (B, C,(T), W, H)

        if num_dim == 4:
            activation_key = activation_key.permute([0,2,3,1]).contiguous()   # (B, W, H, C)
        else:
            assert num_dim == 5
            activation_key = activation_key.permute([0,2,3,4,1]).contiguous()   # (B, T, W, H, C)
        activation_key = activation_key.view(B, -1, self.C)  # (B, (T)WH, C)
        attention_logit = torch.bmm(activation_key, class_key) / np.sqrt(self.C)
        attention = softmax(attention_logit, dim=1)  # (B, (T)WH, K)
        attention = attention.transpose(1, 2)  # (B, K, (T)WH)

        return attention

    def forward(self, x):
        B = x.size()[0]
        num_dim = len(x.size())
        attention = self.compute_attention(x)
        activation_value = self.activation_value_conv(x)  # (B, C, (T), W, H)

        if num_dim == 4:
            activation_value = activation_value.permute([0,2,3,1]).contiguous()  # (B, W, H, C)
        else:
            assert num_dim == 5
            activation_value = activation_value.permute([0,2,3,4,1]).contiguous()   # (B, T, W, H, C)

        activation_value = activation_value.view(B, -1, self.C)  # (B, (T)WH, C)
        attended = torch.bmm(attention, activation_value)  # (B, K, C)

        class_value = self.class_value.expand(B, -1, -1) #(B, C, K)
        class_value = class_value.transpose(1, 2)  # (B, K, C)
        logit = torch.mul(class_value, attended)  # (B, K, C)
        logit = logit.sum(-1)  # (B, K)

        hidden = attended.view(B, -1)
        return logit, hidden
