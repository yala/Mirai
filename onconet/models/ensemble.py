import torch
from torch import nn
from onconet.models.factory import RegisterModel


@RegisterModel("ensemble")
class ModelEnsemble(nn.Module):
    def __init__(self, args):
        '''
            Given a list of ensemble snapshots (in args.ensemble_paths)
            Builds a model that predicts by the mean of all models.
        '''
        super(ModelEnsemble, self).__init__()
        self._models = [torch.load(path) for path in args.ensemble_paths]
        for m in self._models:
            for p in m.parameters():
                p.requires_grad = False

        self.args = args

    def forward(self, x):
        '''
            param x: a batch of image tensors
            returns logit:  mean result over all models
        '''
        logits = [m(x) for m in self._models]

        logit = torch.mean(
            torch.cat([l.unsqueeze(-1) for l in logits], dim=-1), dim=-1)

        return logit
