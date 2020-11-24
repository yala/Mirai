import torch.nn as nn
from onconet.models.factory import RegisterModel
from onconet.datasets.abstract_onco_dataset import DEVICE_TO_ID

NUM_DEVICES = len(set(DEVICE_TO_ID.values()))

@RegisterModel('cross_ent_discriminator')
class Discriminator(nn.Module):
    '''
        Simple MLP discriminator
    '''

    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        num_logits = args.num_classes if not args.survival_analysis_setup else args.max_followup
        if self.args.adv_on_logits_alone:
            self.fc1 = nn.Linear(num_logits, NUM_DEVICES)
        else:
            if self.args.use_risk_factors:
                hidden_dim = args.img_only_dim
            else:
                hidden_dim = args.hidden_dim
            self.fc1 = nn.Linear(hidden_dim + num_logits, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, NUM_DEVICES)
            self.relu = nn.ReLU()


    def forward(self, x):
        if self.args.adv_on_logits_alone:
            return self.fc1(x)
        else:
            hidden = self.relu( self.bn1( self.fc1(x) ))
            hidden = self.relu( self.bn2( self.fc2(hidden) ))
            z = self.fc3( hidden)
            return z
