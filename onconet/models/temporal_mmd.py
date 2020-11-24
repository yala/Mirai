import torch
import numpy as np
import torch.nn as nn
from onconet.models.factory import RegisterModel
from onconet.models.mmd import compute_pairwise_distances, gaussian_kernel
import pdb

SIGMAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
            1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6]



@RegisterModel("temporal_mmd_discriminator")
class TemporalMMDDiscriminator(nn.Module):
    '''
        Simple MMD discriminator with temporal decay. Basic MMD Implementation adapted from: https://github.com/jiangfeng1124/transfer
        with new modifaction to incorperate temporal decay and LIFO cache.
    '''

    def __init__(self, args):
        super(TemporalMMDDiscriminator, self).__init__()
        self.args = args
        num_logits = args.num_classes if not args.survival_analysis_setup else args.max_followup
        assert self.args.use_mmd_adv
        self.sigmas = torch.nn.Parameter(torch.FloatTensor(SIGMAS), requires_grad=False)
        self.x_lifo_cache = {'ages':[], 'entries':[]}
        self.y_lifo_cache = {'ages':[], 'entries':[]}
        self.placeholder = nn.Linear(1,1)


    def update_x_cache(self, x):
        return self.update_cache(self.x_lifo_cache, x)

    def update_y_cache(self, y):
        return self.update_cache(self.y_lifo_cache, y)

    def update_cache(self, cache, z):
        '''
            Update running lifo cache, and cache ages with new tensors.
            args:
            - cache: tuple of ages, and cached tensors
            - z: n * d tensor, consisting of n examples to cache.
            returns:
            - Z: m * d tensor of examples, where m <= args.temporal_mmd_cache_size
            - ages: m lenth list of cache ages
        '''
        assert len(cache['ages']) == len(cache['entries'])
        B, d = z.size()
        new_entries = list(torch.chunk(z, B, dim=0))
        new_ages = np.zeros(B)

        # Remove history from old cache and update ages
        if len(cache['ages']) > 0:
            cache['ages'] += 1
            cache['entries'] = [entry.detach() for entry in cache['entries']]

        # Add new entries
        cache['ages'] = np.concatenate( [cache['ages'], new_ages], axis=0)
        cache['entries'].extend( new_entries )

        # Prune cache to max size
        cache['ages'] = cache['ages'][-self.args.temporal_mmd_cache_size:]
        cache['entries'] = cache['entries'][-self.args.temporal_mmd_cache_size:]

        # combine into one tensor
        Z = torch.cat(cache['entries'], dim=0)
        ages = cache['ages']
        return Z, ages

    def moving_mean(self, distances, x_ages, y_ages):
        '''
        Given distances, compute age-discounted average. Discount computed as args.temporal_decay_factor ** ( x_age + y_age)

        args:
        distances: matrix of pair-wise distances between X and Y (n by m)
        x_ages: length n list of ages of X entries.
        y_ages: length m list of ages of Y entries.

        returns:
        moving_average = Average distance with temporal discounting.
        '''
        n, m = distances.size()
        assert n == len(x_ages)
        assert m == len(y_ages)

        x_age_matrix = np.repeat(x_ages.reshape([n,1]), m, axis=1)
        y_age_matrix = np.repeat(y_ages.reshape([1,m]), n, axis=0)
        age_matrix = x_age_matrix + y_age_matrix

        discount_matrix = np.power( self.args.temporal_mmd_discount_factor, age_matrix)
        discount_matrix = torch.Tensor( discount_matrix).to(self.args.device)
        moving_average = torch.sum(distances * discount_matrix) / torch.sum(discount_matrix)
        return moving_average

    def forward(self, x, y):
        """ Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
        Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
        the distributions of x and y. Here we use kernel two sample estimate
        using the empirical mean of the two distributions.
        Args:
          x: a tensor of shape [num_samples, num_features]
          y: a tensor of shape [num_samples, num_features]
          kernel: a function which computes the kernel in MMD. Defaults to the
          GaussianKernelMatrix.
        Returns:
          a scalar denoting the squared maximum mean discrepancy loss.
        """
        x_with_mem, x_ages = self.update_x_cache(x)
        y_with_mem, y_ages = self.update_y_cache(y)

        cost = self.moving_mean(gaussian_kernel(x_with_mem, x_with_mem, self.sigmas), x_ages, x_ages)
        cost += self.moving_mean(gaussian_kernel(y_with_mem, y_with_mem, self.sigmas), y_ages, y_ages)
        cost -= 2 * self.moving_mean(gaussian_kernel(x_with_mem, y_with_mem, self.sigmas), x_ages, y_ages)

        cost = torch.clamp(cost, min=0)
        return cost

