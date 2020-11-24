import torch
import torch.nn as nn
import pdb



class Cumulative_Probability_Layer(nn.Module):
    def __init__(self, num_features, args, max_followup):
        super(Cumulative_Probability_Layer, self).__init__()
        self.args = args
        self.hazard_fc = nn.Linear(num_features,  max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu = nn.ReLU(inplace=True)
        mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter('upper_triagular_mask', mask)

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.relu(raw_hazard)
        return pos_hazard

    def forward(self, x):
        if self.args.make_probs_indep:
            return self.hazards(x)
#        hazards = self.hazard_fc(x)
        hazards = self.hazards(x)
        B, T = hazards.size() #hazards is (B, T)
        expanded_hazards = hazards.unsqueeze(-1).expand(B, T, T) #expanded_hazards is (B,T, T)
        masked_hazards = expanded_hazards * self.upper_triagular_mask # masked_hazards now (B,T, T)
        cum_prob = torch.sum(masked_hazards, dim=1) + self.base_hazard_fc(x)
        return cum_prob
