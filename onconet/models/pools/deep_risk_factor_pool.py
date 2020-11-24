import torch
import torch.nn as nn
from onconet.models.pools.abstract_pool import AbstractPool
from onconet.models.pools.factory import RegisterPool
from onconet.models.pools.factory import get_pool
from onconet.utils.risk_factors import RiskFactorVectorizer
import pdb

@RegisterPool('DeepRiskFactorPool')
class RiskFactorPool(AbstractPool):
    def __init__(self, args, num_chan):
        super(RiskFactorPool, self).__init__(args, num_chan)
        self.args = args
        self.internal_pool = get_pool(args.pool_name)(args, num_chan)
        assert not self.internal_pool.replaces_fc()
        assert not args.pred_risk_factors
        self.length_risk_factor_vector = RiskFactorVectorizer(args).vector_length

        input_dim = self.length_risk_factor_vector + num_chan

        self.fc1 = nn.Linear(input_dim, num_chan)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(num_chan)
        self.dropout = nn.Dropout(args.dropout)
        self.fc2 = nn.Linear(num_chan, args.num_classes)
        self.args.hidden_dim = num_chan

    def replaces_fc(self):
        return True

    def forward(self, x, risk_factors):
        if self.args.replace_snapshot_pool:
            x = x.data
            if self.args.cuda:
                x = x.cuda()
        _, image_features = self.internal_pool(x)
        hidden = torch.cat((image_features, torch.cat(risk_factors)), 1)
        hidden = self.fc1(hidden)
        hidden = self.relu(hidden)
        hidden = self.bn(hidden)
        hidden = self.dropout(hidden)
        logit = self.fc2(hidden)
        return logit, hidden
