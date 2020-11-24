import numpy as np
import math
import sklearn.metrics
import torch
import torch.nn.functional as F
import pdb
from onconet.utils.region_annotation import get_annotation_mask

NEED_TARGET_X_FOR_STEP_ERR = "Target x must be defined for {} step"
UNRECOG_REGION_LOSS_TYPE_ERR = "Region loss type {} not recognized!"

def get_model_loss(logit, y, batch, args):
    if args.survival_analysis_setup:
        if args.pred_both_sides:
            loss = 0
            for side in 'l','r':
                y_seq, y_mask = batch['y_seq_{}'.format(side)], batch['y_mask_{}'.format(side)]
                loss += F.binary_cross_entropy_with_logits(logit[side], y_seq.float(), weight=y_mask.float(), size_average=False)/ torch.sum(y_mask.float())
        else:
            y_seq, y_mask = batch['y_seq'], batch['y_mask']
            loss = F.binary_cross_entropy_with_logits(logit, y_seq.float(), weight=y_mask.float(), size_average=False)/ torch.sum(y_mask.float())
    elif args.eval_survival_on_risk:
        loss = F.binary_cross_entropy_with_logits(logit, y.float())
    elif args.objective == 'cross_entropy':
        loss = F.cross_entropy(logit, y)
    else:
        raise Exception(
            "Objective {} not supported!".format(args.objective))
    return loss

def get_birads_loss(activ_dict, model, batch, args):
    birads_logit = activ_dict['birads_logit']
    birads_y = batch['birads']
    birads_y.to(args.device)
    birads_loss = F.cross_entropy(birads_logit, birads_y)
    return birads_loss

def get_region_loss(activ_dict, logit, batch, train_model, args):
    '''
        - activ: Model activations,  comes in shape [B, C, (T), H, W]. Note H,W here are after network downsampling
        - logit: logits for predictions, come in shape [B, num_classes]
        - model: primary model to get out predictions. Must be something ontop of resnet base
        - batch: batch from an abstract onco-object
        - train_model: whether need to do bo backprop after
        - args: runtime args

        returns: region loss per args.
    '''
    volatile = not train_model
    region_loss = torch.zeros([])
    region_loss = region_loss.to(args.device)

    activ = activ_dict['activ']
    region_logits = activ_dict['region_logit']

    num_dim = len(activ.size())
    if num_dim == 4:
        B, C, H, W = activ.size()
    else:
        assert num_dim == 5
        B, C, T, H, W = activ.size()

    epsilon = 1e-8
    region_loss_type = args.region_annotation_loss_type
    has_region_defined = batch['region_bottom_left_x'] >= 0
    region_annotation_mask = get_annotation_mask(activ, batch, volatile, args) #[B, (T), H, W]

    any_region_defined = has_region_defined.max().data.item() == 1 and torch.sum(region_annotation_mask).data.item() > 0

    if not any_region_defined:
        return region_loss

    has_region_defined.to(args.device)
    if num_dim == 4:
        sample_mask = has_region_defined.expand([ H*W, B]).transpose(0,1).float()
    else:
        sample_mask = has_region_defined.expand([ H*W, B, T]).permute([1,2,0]).contiguous().view([B,-1]).float()

    flat_region_annotation_mask = region_annotation_mask.view([B, -1]).float()
    flat_region_annotation_mask = flat_region_annotation_mask / (torch.sum(flat_region_annotation_mask, dim=1).unsqueeze(-1) + epsilon)

    if region_loss_type == 'pred_region':
        region_preds = F.sigmoid(region_logits)
        target_probs =  region_preds * (region_annotation_mask == 1).float() + (1 - region_preds) * (region_annotation_mask == 0).float()
        focal_loss_weighting = (1 - target_probs)**args.region_annotation_focal_loss_lambda if args.region_annotation_focal_loss_lambda > 0 else 1
        masked_focal_loss_weighting = focal_loss_weighting * sample_mask.view_as(target_probs).float()
        focal_loss = F.binary_cross_entropy_with_logits(region_logits, region_annotation_mask.float(), weight=masked_focal_loss_weighting.detach() , size_average=False)
        region_loss =  focal_loss / (torch.sum(region_annotation_mask.float()) + epsilon)

    else:
        raise NotImplementedError(UNRECOG_REG_LOSS_TYPE_ERR.format(region_loss_type))

    return region_loss

def model_step(x, y, risk_factors, batch, models, optimizers, train_model,  args):
    '''
        Single step of running model on the a batch x,y and computing
        the loss. Backward pass is computed if train_model=True.

        Returns various stats of this single forward and backward pass.


        args:
        - x: input features
        - y: labels
        - risk_factors: additional input features corresponding to risk factors
        - batch: whole batch dict, can be used by various special args
        - models: dict of models. The main model, named "model" must return logit, hidden, activ
        - optimizers: dict of optimizers for models
        - train_model: whether or not to compute backward on loss
        - args: various runtime args such as batch_split etc
        returns:
        - loss: scalar for loss on batch as a tensor
        - reg_loss: scalar for regularization loss on batch as a tensor
        - preds: predicted labels as numpy array
        - probs: softmax probablities as numpy array
        - golds: labels, numpy array version of arg y
        - exams: exam ids for batch if available
        - hiddens: feature rep for batch
    '''
    if args.use_risk_factors:
        logit, hidden, activ_dict = models['model'](x, risk_factors=risk_factors, batch=batch)
    else:
        logit, hidden, activ_dict = models['model'](x, batch=batch)

    if args.downsample_activ:
        assert not args.use_precomputed_hiddens
        activ = activ_dict['activ']
        activ = F.max_pool2d(activ, 12, stride=12)

    if args.eval_survival_on_risk:
        logit = logit[:, args.years_risk - 1]

    loss = get_model_loss(logit, y, batch, args)

    if args.pred_both_sides:
        logit, _ = torch.max( torch.cat( [logit['l'].unsqueeze(-1), logit['r'].unsqueeze(-1)], dim=-1), dim=-1)

    reg_loss = 0
    adv_loss = 0

    if args.use_region_annotation:
        assert not args.use_precomputed_hiddens
        region_loss = get_region_loss(activ_dict, logit, batch, train_model, args)
        loss += args.regularization_lambda * region_loss
        reg_loss += args.regularization_lambda * region_loss

    if args.predict_birads:
        birads_loss = get_birads_loss(activ_dict, models['model'], batch, args)
        loss += args.predict_birads_lambda * birads_loss
        reg_loss += args.predict_birads_lambda * birads_loss

    if args.pred_risk_factors and 'pred_rf_loss' in activ_dict:
        pred_rf_loss = activ_dict['pred_rf_loss']
        if args.data_parallel:
            pred_rf_loss = torch.mean(pred_rf_loss)
        loss += args.pred_risk_factors_lambda * pred_rf_loss
        reg_loss += args.pred_risk_factors_lambda * pred_rf_loss

    if args.pred_missing_mammos:
        pred_masked_mammo_loss = activ_dict['pred_masked_mammo_loss'].mean() if args.data_parallel else activ_dict['pred_masked_mammo_loss']
        loss += args.pred_missing_mammos_lambda * pred_masked_mammo_loss
        reg_loss += args.pred_missing_mammos_lambda * pred_masked_mammo_loss

    if args.use_adv:
        gen_loss, adv_loss  = get_adv_loss(models, hidden, logit, batch, args)
        # Noew train discrim til loss below device entropy across train set
        train_adv = not args.use_mmd_adv and train_model and (args.step_indx != 0 or not  args.train_adv_seperate)
        if train_adv :
            num_adv_steps = 0
            adv_loss = adv_step(hidden, logit, batch, models, optimizers, train_model,  args)
            while adv_loss > args.device_entropy and num_adv_steps < args.num_adv_steps and not args.train_adv_seperate:
                num_adv_steps += 1
                adv_loss = adv_step(hidden, logit, batch, models, optimizers, train_model,  args)

        if args.anneal_adv_loss:
            args.curr_adv_lambda +=  args.adv_loss_lambda/ 10000
            adv_loss_lambda = min( args.curr_adv_lambda, args.adv_loss_lambda)
        else:
            adv_loss_lambda = args.adv_loss_lambda

        loss += adv_loss_lambda * gen_loss
        reg_loss += adv_loss_lambda* gen_loss

        if args.turn_off_model_train or (train_model and args.step_indx != 0 and args.train_adv_seperate):
            loss *= 0
            reg_loss *= 0

    loss /= args.batch_splits
    adv_loss /= args.batch_splits
    reg_loss /= args.batch_splits

    if train_model:
        args.step_indx =  (args.step_indx + 1) % (args.num_adv_steps+1)
        loss.backward()

    if args.survival_analysis_setup or args.eval_survival_on_risk:
        probs = F.sigmoid(logit).cpu().data.numpy() #Shape is B, Max followup
        preds = probs > .5
    else:
        batch_softmax = F.softmax(logit, dim=-1).cpu()
        preds = torch.max(batch_softmax, 1)[1].view(y.size()).data.numpy()
        probs = batch_softmax[:,1].data.numpy().tolist()
    golds = y.data.cpu().numpy()

    exams = batch['exam'] if 'exam' in batch else None
    if args.get_activs_instead_of_hiddens:
        hiddens = activ.data.cpu().numpy()
    else:
        hiddens = hidden.data.cpu().numpy()


    return  loss, reg_loss, preds, probs, golds, exams, hiddens, adv_loss

def get_adv_loss(models, hidden, logit,  batch, args):
    if args.use_mmd_adv:
        return get_mmd_loss(models, hidden, logit,  batch, args)
    else:
        return get_cross_entropy_adv_loss(models, hidden, logit,  batch, args)

def get_cross_entropy_adv_loss(models, hidden, logit,  batch, args):
    '''
        Return generator, and adversary loss according to ability of advesary to disnguish device
        , as defined in https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf.
        Idea is based on domain classifier
        args:
        - models: dict of avaialble models. adv must be defined
        - hidden: hidden representation of generated distribution
        - logit: score distrubution given those hiddens
        - batch: full batch dict
        - args: run time args
        returns:
        - gen_loss: loss to update generator, or in most cases, network to regularize
        - adv_loss: loss to update adversary.
    '''
    adv = models['adv']
    img_hidden = hidden[:,:args.img_only_dim] if args.use_risk_factors else hidden
    img_only_dim = img_hidden.size()[-1]
    device, device_known, y = batch['device'], batch['device_is_known'].float(), batch['y']
    if args.use_precomputed_hiddens or args.model_name == 'mirai_full':
        B, N, _ = hidden.size()
        _, C = logit.size()
        y = y.unsqueeze(0).transpose(0,1).expand_as(device)
        logit = logit.unsqueeze(1).expand([B,N,C]).contiguous().view([-1, C])
        y = y.contiguous().view(-1)
        device = device.view(-1)
        device_known = device_known.view( -1)
        img_hidden = img_hidden.view( [-1, img_only_dim])

    if args.adv_on_logits_alone:
        device_logit = adv(logit)
    else:
        device_logit = adv(torch.cat([img_hidden, logit.detach()], dim=1))

    adv_loss_per_sample = F.cross_entropy(device_logit, device, reduce=False) * device_known
    adv_loss = torch.sum(adv_loss_per_sample) / (torch.sum(device_known) + 1e-6)
    gen_loss = -adv_loss
    return gen_loss, adv_loss

def get_mmd_loss(models, hidden, logit,  batch, args):
    '''
        Return generator, and adversary loss according to ability of advesary to disnguish device
        by MMD based discriminator. Align by class the older device image hiddens (2 or 1) to new device (0)
        hidden representation
        args:
        - models: dict of avaialble models. adv must be defined
        - hidden: hidden representation of generated distribution
        - logit: score distrubution given those hiddens
        - batch: full batch dict
        - args: run time args
        returns:
        - gen_loss: loss to update generator, or in most cases, network to regularize
        - adv_loss: loss to update adversary. not used for MMD since MMD is non-parametric
    '''
    pos_adv = models['pos_adv']
    neg_adv = models['neg_adv']

    img_hidden = hidden[:,:args.img_only_dim] if args.use_risk_factors and not args.use_precomputed_hiddens  else hidden
    img_only_dim = img_hidden.size()[-1]
    device, device_known, y = batch['device'], batch['device_is_known'], batch['y']

    if args.use_precomputed_hiddens:
        y = y.unsqueeze(0).transpose(0,1).expand_as(device)
        y = y.contiguous().view(-1)
        device = device.view(-1)
        device_known = device_known.view( -1)
        img_hidden = img_hidden.view( [-1, img_only_dim])

    is_pos = y == 1
    is_neg = y == 0
    is_source_device = (device == 0) * (device_known == 1)
    source_pos_hidden = torch.masked_select(img_hidden, (is_source_device * is_pos).unsqueeze(-1)).view(-1, img_only_dim)
    source_neg_hidden = torch.masked_select(img_hidden, (is_source_device * is_neg).unsqueeze(-1)).view(-1, img_only_dim)

    is_target_device = (device == 2) * (device_known == 1)
    target_pos_hidden = torch.masked_select(img_hidden, (is_target_device * is_pos).unsqueeze(-1)).view(-1, img_only_dim)
    target_neg_hidden = torch.masked_select(img_hidden, (is_target_device * is_neg).unsqueeze(-1)).view(-1, img_only_dim)

    pos_hidden = torch.masked_select(img_hidden, is_pos.unsqueeze(-1)).view(-1, img_only_dim)
    neg_hidden = torch.masked_select(img_hidden, is_neg.unsqueeze(-1)).view(-1, img_only_dim)

    if source_pos_hidden.nelement() > 0 and target_pos_hidden.nelement() > 0:
        pos_mmd = pos_adv(source_pos_hidden, target_pos_hidden).view([])
    else:
        pos_mmd = 0

    if source_neg_hidden.nelement() > 0 and target_neg_hidden.nelement() > 0:
        neg_mmd = neg_adv(source_neg_hidden, target_neg_hidden).view([])
    else:
        neg_mmd = 0

    gen_loss = pos_mmd + neg_mmd

    if args.add_repulsive_mmd and pos_hidden.nelement() > 0 and neg_hidden.nelement() > 0:
        repl_adv = models['repel_adv']
        replusive_mmd = repl_adv(pos_hidden, neg_hidden)
        gen_loss -= 2 * replusive_mmd

    adv_loss = -gen_loss
    return gen_loss, adv_loss

def adv_step(hidden, logit, batch, models, optimizers, train_model,  args):
    '''
        Single step of running kl adversary on the a batch x,y and computing
        its loss. Backward pass is computed if train_model=True.
        Returns loss
        args:
        - hidden: hidden features
        - logit: estimate of posterior
        - batch: whole batch dict, can be used by various special args
        - models: dict of models. The main model, named "model" must return logit, hidden, activ
        - train_model: whether or not to compute backward on loss
        - args: various runtime args such as batch_split etc
        returns:
        - losses, a dict of losses containing 'klgan' with the kl adversary loss
    '''

    hidden_with_no_hist, logit_no_hist = hidden.detach(), logit.detach()

    _, adv_loss = get_adv_loss(models, hidden_with_no_hist, logit_no_hist, batch, args)
    if train_model:
        adv_loss.backward(retain_graph=True)
        optimizers['adv'].step()
        optimizers['adv'].zero_grad()
    return  adv_loss

