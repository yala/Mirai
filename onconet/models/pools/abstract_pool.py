from abc import ABCMeta, abstractmethod
import torch.nn as nn


class AbstractPool(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, args, num_chan):
        super(AbstractPool, self).__init__()

    @property
    @abstractmethod
    def replaces_fc(self):
        pass
