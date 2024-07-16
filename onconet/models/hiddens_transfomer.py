
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from onconet.models.pools.factory import get_pool
from onconet.models.factory import RegisterModel
from onconet.models.cumulative_probability_layer import Cumulative_Probability_Layer

EMBEDDING_DIM = 96
MAX_TIME = 10
MAX_VIEWS = 2
MAX_SIDES = 2

@RegisterModel("transformer")
class AllImageTransformer(nn.Module):
    def __init__(self, args):

        super(AllImageTransformer, self).__init__()

        self.args = args
        self.args.wrap_model = False
        args.hidden_dim = args.transfomer_hidden_dim
        assert args.use_precomputed_hiddens or args.model_name == 'mirai_full'

        self.projection_layer = nn.Linear(args.precomputed_hidden_dim, args.hidden_dim)
        self.mask_embedding = torch.nn.Embedding(2, args.precomputed_hidden_dim, padding_idx=1)
        kept_images_vec = torch.nn.Parameter(torch.ones([1,args.num_images,1]),
            requires_grad=False)
        self.register_parameter('kept_images_vec', kept_images_vec)
        self.transformer = Transformer(args)

        self.pred_masked_img_fc = nn.Linear(args.hidden_dim, args.precomputed_hidden_dim)
        pool_name = args.pool_name
        if args.use_risk_factors:
            pool_name = 'DeepRiskFactorPool' if self.args.deep_risk_factor_pool else 'RiskFactorPool'
        self.pool = get_pool(pool_name)(args, args.hidden_dim)


        if not self.pool.replaces_fc():
            # Cannot not placed on self.all_blocks since requires intermediate op
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=args.dropout)
            self.fc = nn.Linear(args.hidden_dim, args.num_classes)

        if args.survival_analysis_setup:
            if args.pred_both_sides:
                self.prob_of_failure_layer_l = Cumulative_Probability_Layer(args.hidden_dim, args, max_followup=args.max_followup)
                self.prob_of_failure_layer_r = Cumulative_Probability_Layer(args.hidden_dim, args, max_followup=args.max_followup)
            else:
                self.prob_of_failure_layer = Cumulative_Probability_Layer(args.hidden_dim, args, max_followup=args.max_followup)

    def mask_input(self, x, view_seq):
        B, N, _ = x.size()
        mask_prob = self.args.mask_prob if self.training and self.args.pred_missing_mammos else 0
        is_mask = torch.bernoulli( self.kept_images_vec.expand([B,N,1]) * mask_prob).to(x.device) #0 is not masked, 1 is masked
        # Don't mask out any PAD images
        is_mask = is_mask * (view_seq < MAX_VIEWS).unsqueeze(-1).float() # sets all not viewed as 0
        is_kept = 1 - is_mask
        x = x * is_kept + self.mask_embedding(is_kept.squeeze(-1).long())
        if self.args.also_pred_given_mammos:
            # all set to all ones so all are predicted
            is_mask = (is_mask >= -1).float() * (view_seq < MAX_VIEWS).unsqueeze(-1).float()
        return x, is_mask

    def get_pred_mask_loss(self, transformer_hidden, x, is_mask):
        if is_mask.sum().item() == 0:
            return 0
        B, N, D_n = transformer_hidden.size()
        _, _, D_o = x.size()
        hidden_for_mask = torch.masked_select(transformer_hidden, is_mask.byte()).view(-1, D_n)
        pred_x = self.pred_masked_img_fc(hidden_for_mask)
        x_for_mask = torch.masked_select(x, is_mask.byte()).view(-1, D_o)
        return F.mse_loss(pred_x, x_for_mask)

    def forward(self, x, risk_factors=None, batch=None):
        """Computes a forward pass of the model.

        Arguments:
            x(Variable): The input to the model.

        Returns:
            The result of feeding the input through the model.
        """
        # Go through all layers up to fc
        time_seq, view_seq, side_seq = batch['time_seq'], batch['view_seq'], batch['side_seq']
        masked_x, is_mask = self.mask_input(x, view_seq)
        masked_x = self.projection_layer(masked_x)
        transformer_hidden = self.transformer(masked_x, time_seq, view_seq, side_seq)

        img_like_hidden = transformer_hidden.transpose(1,2).unsqueeze(-1)
        logit, hidden = self.aggregate_and_classify(img_like_hidden, risk_factors=risk_factors)

        activ_dict = {}
        try:
            if self.args.predict_birads:
                activ_dict['birads_logit'] = self.birads_fc(hidden)
            if self.args.pred_risk_factors:
                activ_dict['pred_rf_loss'] = self.pool.get_pred_rf_loss(hidden, risk_factors)

            if self.args.pred_missing_mammos:
                activ_dict['pred_masked_mammo_loss'] = self.get_pred_mask_loss(transformer_hidden, x, is_mask)
        except:
            pass
        return logit, transformer_hidden, activ_dict


    def aggregate_and_classify(self, x, risk_factors=None):
        # Pooling layer
        if self.args.use_risk_factors:
            logit, hidden = self.pool(x, risk_factors)
        else:
            logit, hidden = self.pool(x)

        if not self.pool.replaces_fc():
            # self.fc is always on last gpu, so direct call of fc(x) is safe
            try:
                # placed in try catch for back compatbility.
                hidden = self.relu(hidden)
            except :
                pass
            hidden = self.dropout(hidden)
            logit = self.fc(hidden)

        if self.args.survival_analysis_setup:
            if self.args.pred_both_sides:
                logit = {'l':self.prob_of_failure_layer_l(hidden) , 'r':self.prob_of_failure_layer_r(hidden)}
            else:
                logit = self.prob_of_failure_layer(hidden)

        return logit, hidden



class Transformer(nn.Module):
    def __init__(self, args):

        super(Transformer, self).__init__()

        self.args = args
        self.args.wrap_model = False
        assert EMBEDDING_DIM % 3 == 0
        self.time_embed =  torch.nn.Embedding(MAX_TIME+1, EMBEDDING_DIM//3, padding_idx=-1)
        self.view_embed = torch.nn.Embedding(MAX_VIEWS+1, EMBEDDING_DIM//3 , padding_idx=-1)
        self.side_embed = torch.nn.Embedding(MAX_SIDES+1, EMBEDDING_DIM//3 , padding_idx=-1)
        self.embed_add_fc = nn.Linear(EMBEDDING_DIM, args.hidden_dim)
        self.embed_scale_fc = nn.Linear(EMBEDDING_DIM, args.hidden_dim)
        for layer in range(args.num_layers):
            transformer_layer = TransformerLayer(args)
            self.add_module('transformer_layer_{}'.format(layer), transformer_layer)

    def condition_on_pos_embed(self, x, embed):
        return self.embed_scale_fc(embed) * x + self.embed_add_fc(embed)

    def forward(self, x, time_seq, view_seq, side_seq):
        """Computes a forward pass of the model.

        Arguments:
            x(Variable): The input to the model.

        Returns:
            The result of feeding the input through the model.
        """
        # Add positional embeddings
        view, time, side = self.view_embed(view_seq), self.time_embed(time_seq), self.side_embed(side_seq)
        embed = torch.cat( [view, time, side], dim=-1)
        x = self.condition_on_pos_embed(x, embed)

        # Run through transformer
        for indx in range(self.args.num_layers):
            name = 'transformer_layer_{}'.format(indx)
            x = self._modules[name](x)

        return x


class TransformerLayer(nn.Module):
    def __init__(self, args):
        super(TransformerLayer, self).__init__()

        self.args = args
        self.multihead_attention = MultiHead_Attention(self.args)
        self.layernorm_attn = nn.LayerNorm(self.args.hidden_dim)
        self.fc1 = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.layernorm_fc = nn.LayerNorm(self.args.hidden_dim)

    def forward(self, x):
        h = self.multihead_attention(x)
        x = self.layernorm_attn( h + x)
        h = self.fc2(self.relu(self.fc1(x)))
        x = self.layernorm_fc(h + x)
        return x

class MultiHead_Attention(nn.Module):
    def __init__(self, args):
        super(MultiHead_Attention, self).__init__()
        self.args = args
        assert args.hidden_dim % args.num_heads == 0

        self.query = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.value = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.key = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.dropout = nn.Dropout(p=args.dropout)

        self.dim_per_head = args.hidden_dim // args.num_heads

        self.aggregate_fc = nn.Linear(args.hidden_dim, args.hidden_dim)


    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.dim_per_head)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, x):
        B, N, H = x.size()

        # perform linear operation and split into h heads

        k = self.key(x).view(B, N, self.args.num_heads, self.dim_per_head)
        q = self.query(x).view(B, N, self.args.num_heads, self.dim_per_head)
        v = self.value(x).view(B, N, self.args.num_heads, self.dim_per_head)

        # transpose to get dimensions B * args.num_heads * S * dim_per_head
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        h = self.attention(q, k, v)

        # concatenate heads and put through final linear layer
        h = h.transpose(1,2).contiguous().view(B, -1, H)

        output = self.aggregate_fc(h)

        return output


