import torch
import torch.nn as nn
from onconet.models.factory import RegisterModel
import pdb

SIGMAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
            1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6]




def compute_pairwise_distances(x, y):
    """ Computes the squared pairwise Euclidean distances between x and y.
    Args:
      x: a tensor of shape [num_x_samples, num_features]
      y: a tensor of shape [num_y_samples, num_features]
    Returns:
      a distance matrix of dimensions [num_x_samples, num_y_samples]
    Raise:
      ValueError: if the inputs do no matched the specified dimensions.
    """

    if not len(x.size()) == len(y.size()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.size()[1] != y.size()[1]:
        raise ValueError('The number of features should be the same.')

    # By making the `inner' dimensions of the two matrices equal to 1 using
    # broadcasting then we are essentially substracting every pair of rows
    # of x and y.
    norm = lambda x: torch.sum(x * x, 1)
    return norm(x.unsqueeze(2) - y.t())

def gaussian_kernel(x, y, sigmas):
    """ Computes a Gaussian RBK between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      sigmas: a tensor of floats which denote the widths of each of the
        gaussians in the kernel.
    Returns:
      A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel
    """
    beta = 1. / (2. * (sigmas.unsqueeze(1)))

    dist = compute_pairwise_distances(x, y)

    s = torch.matmul(beta, dist.view(1, -1))
    return (torch.sum(torch.exp(-s), 0)).view_as(dist)


@RegisterModel("mmd_discriminator")
class MMDDiscriminator(nn.Module):
    '''
        Simple MMD discriminator. Implementation adapted from: https://github.com/jiangfeng1124/transfer
    '''

    def __init__(self, args):
        super(MMDDiscriminator, self).__init__()
        self.args = args
        num_logits = args.num_classes if not args.survival_analysis_setup else args.max_followup
        assert self.args.use_mmd_adv
        self.sigmas = torch.nn.Parameter(torch.FloatTensor(SIGMAS), requires_grad=False)
        self.placeholder = nn.Linear(1,1)

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

        cost = torch.mean(gaussian_kernel(x, x, self.sigmas))
        cost += torch.mean(gaussian_kernel(y, y, self.sigmas))
        cost -= 2 * torch.mean(gaussian_kernel(x, y, self.sigmas))

        cost = torch.clamp(cost, min=0)
        return cost

