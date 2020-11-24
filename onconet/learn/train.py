import os
import math
import numpy as np
import sklearn.metrics
import torch
from tqdm import tqdm
import onconet.models.factory as model_factory
import onconet.learn.state_keeper as state
import onconet.utils.stats as stats
from onconet.learn.utils import cluster_results_by_exam, ignore_None_collate, init_metrics_dictionary, \
    get_train_and_dev_dataset_loaders, compute_eval_metrics, \
    get_human_preds
from onconet.learn.step import model_step, adv_step
import warnings
import pdb

tqdm.monitor_interval=0

NUM_RESAMPLES_DURING_TRAIN = 100

def get_train_variables(args, model):
    '''
        Given args, and whether or not resuming training, return
        relevant train variales.

        returns:
        - start_epoch:  Index of initial epoch
        - epoch_stats: Dict summarizing epoch by epoch results
        - state_keeper: Object responsibile for saving and restoring training state
        - batch_size: sampling batch_size
        - models: Dict of models
        - optimizers: Dict of optimizers, one for each model
        - tuning_key: Name of epoch_stats key to control learning rate by
        - num_epoch_sans_improvement: Number of epochs since last dev improvment, as measured by tuning_key
        - num_epoch_since_reducing_lr: Number of epochs since last lr reduction
        - no_tuning_on_dev: True when training does not adapt based on dev performance
    '''
    start_epoch = 1
    if args.current_epoch is not None:
        start_epoch = args.current_epoch
    if args.lr is None:
        args.lr = args.init_lr
    if args.anneal_adv_loss:
        args.curr_adv_lambda = 0
    if args.epoch_stats is not None:
        epoch_stats = args.epoch_stats
    else:
        epoch_stats = init_metrics_dictionary(modes=['train', 'dev'])

    state_keeper = state.StateKeeper(args)
    batch_size = args.batch_size // args.batch_splits

    # Set up models
    if isinstance(model, dict):
        models = model
    else:
        models = {'model': model }

    if args.use_adv:
        if args.use_mmd_adv:
            adv_name = 'temporal_mmd_discriminator' if args.use_temporal_mmd else 'mmd_discriminator'
            models["pos_adv"] = model_factory.get_model_by_name(adv_name, False, args)
            models["neg_adv"] = model_factory.get_model_by_name(adv_name, False, args)

            if args.add_repulsive_mmd:
                models['repel_adv'] = model_factory.get_model_by_name(adv_name, False, args)
        else:
            adv_name = 'cross_ent_discriminator'
            models["adv"] = model_factory.get_model_by_name(adv_name, False, args)

    # Setup optimizers
    optimizers = {}
    for name in models:
        model = models[name].to(args.device)
        optimizers[name] = model_factory.get_optimizer(model, args)

    if args.optimizer_state is not None:
        for optimizer_name in args.optimizer_state:
            state_dict = args.optimizer_state[optimizer_name]
            optimizers[optimizer_name] = state_keeper.load_optimizer(
                optimizers[optimizer_name],
                state_dict)

    num_epoch_sans_improvement = 0
    num_epoch_since_reducing_lr = 0

    no_tuning_on_dev = args.no_tuning_on_dev or args.ten_fold_cross_val

    tuning_key = "dev_{}".format(args.tuning_metric)

    return start_epoch, epoch_stats, state_keeper, batch_size, models, optimizers, tuning_key, num_epoch_sans_improvement, num_epoch_since_reducing_lr, no_tuning_on_dev


def train_model(train_data, dev_data, model, args):
    '''
        Train model and tune on dev set. If model doesn't improve dev performance within args.patience
        epochs, then halve the learning rate, restore the model to best and continue training.

        At the end of training, the function will restore the model to best dev version.

        returns epoch_stats: a dictionary of epoch level metrics for train and test
        returns models : dict of models, containing best performing model setting from this call to train
    '''

    start_epoch, epoch_stats, state_keeper, batch_size, models, optimizers, tuning_key, num_epoch_sans_improvement, num_epoch_since_reducing_lr, no_tuning_on_dev = get_train_variables(
        args, model)

    train_data_loader, dev_data_loader = get_train_and_dev_dataset_loaders(
        args,
        train_data,
        dev_data,
        batch_size)
    for epoch in range(start_epoch, args.epochs + 1):

        print("-------------\nEpoch {}:\n".format(epoch))

        for mode, data_loader in [('Train', train_data_loader), ('Dev', dev_data_loader)]:
            train_model = mode == 'Train'
            key_prefix = mode.lower()
            loss,  golds, preds, probs, exams, reg_loss, censor_times, adv_loss = run_epoch(
                data_loader,
                train_model=train_model,
                truncate_epoch=True,
                models=models,
                optimizers=optimizers,
                args=args)

            log_statement, epoch_stats = compute_eval_metrics(args, loss, golds, preds,
                                                            probs, exams, reg_loss, censor_times, adv_loss, epoch_stats, key_prefix)

            if mode == 'Dev' and 'mammo_1year' in args.dataset:
                dev_human_preds = get_human_preds(exams, dev_data.metadata_json)
                threshold, _ = stats.get_thresholds_interval(probs, golds, dev_human_preds,
                    rebalance_eval_cancers=args.rebalance_eval_cancers, num_resamples=NUM_RESAMPLES_DURING_TRAIN)
                print(' Dev Threshold: {:.8f} '.format(threshold))
                (fnr, _), (tpr, _), (tnr, _) = stats.get_rates_intervals(probs, golds, threshold,
                                rebalance_eval_cancers=args.rebalance_eval_cancers, num_resamples=NUM_RESAMPLES_DURING_TRAIN)
                epoch_stats['{}_fnr'.format(key_prefix)].append(fnr)
                epoch_stats['{}_tnr'.format(key_prefix)].append(tnr)
                epoch_stats['{}_tpr'.format(key_prefix)].append(tpr)
                log_statement = "{} fnr: {:.3f} tnr: {:.3f} tpr: {:.3f}".format(log_statement, fnr, tnr, tpr)

            print(log_statement)

        # Save model if beats best dev, or if not tuning on dev
        best_func, arg_best = (min, np.argmin) if tuning_key == 'dev_loss' else (max, np.argmax)
        improved = best_func(epoch_stats[tuning_key]) == epoch_stats[tuning_key][-1]
        if improved or no_tuning_on_dev:
            num_epoch_sans_improvement = 0
            if not os.path.isdir(args.save_dir):
                os.makedirs(args.save_dir)
            epoch_stats['best_epoch'] = arg_best( epoch_stats[tuning_key] )
            state_keeper.save(models, optimizers, epoch, args.lr, epoch_stats)

        num_epoch_since_reducing_lr += 1
        if improved:
            num_epoch_sans_improvement = 0
        else:
            num_epoch_sans_improvement += 1
        print('---- Best Dev {} is {} at epoch {}'.format(
            args.tuning_metric,
            epoch_stats[tuning_key][epoch_stats['best_epoch']],
            epoch_stats['best_epoch'] + 1))

        if num_epoch_sans_improvement >= args.patience or \
                (no_tuning_on_dev and num_epoch_since_reducing_lr >= args.lr_reduction_interval):
            print("Reducing learning rate")
            num_epoch_sans_improvement = 0
            num_epoch_since_reducing_lr = 0
            if not args.turn_off_model_reset:
                models, optimizer_states, _, _, _ = state_keeper.load()

                # Reset optimizers
                for name in optimizers:
                    optimizer = optimizers[name]
                    state_dict = optimizer_states[name]
                    optimizers[name] = state_keeper.load_optimizer(optimizer, state_dict)
            # Reduce LR
            for name in optimizers:
                optimizer = optimizers[name]
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay

            # Update lr also in args for resumable usage
            args.lr *= .5

    # Restore model to best dev performance, or last epoch when not tuning on dev
    models, _, _, _, _ = state_keeper.load()

    return epoch_stats, models


def compute_threshold_and_dev_stats(dev_data, models, args):
    '''
    Compute threshold based on the Dev results
    '''
    if not isinstance(models, dict):
        models = {'model': models}
    models['model'] = models['model'].to(args.device)

    dev_stats = init_metrics_dictionary(modes=['dev'])

    batch_size = args.batch_size // args.batch_splits
    data_loader = torch.utils.data.DataLoader(
        dev_data,
        batch_size = batch_size,
        shuffle = False,
        num_workers = args.num_workers,
        collate_fn = ignore_None_collate,
        pin_memory=True,
        drop_last = False)
    loss, golds, preds, probs, exams, reg_loss, censor_times, adv_loss = run_epoch(
        data_loader,
        train_model=False,
        truncate_epoch=False,
        models=models,
        optimizers=None,
        args=args)


    if ('detection' in args.dataset or 'risk' in args.dataset) and '1year' in args.dataset and not args.survival_analysis_setup:
        human_preds = get_human_preds(exams, dev_data.metadata_json)


        threshold, (th_lb, th_ub) = stats.get_thresholds_interval(probs, golds, human_preds, rebalance_eval_cancers=args.rebalance_eval_cancers)
        args.threshold = threshold
        print(' Dev Threshold: {:.8f} ({:.8f} - {:.8f})'.format(threshold, th_lb, th_ub))
    else:
        args.threshold = None
    log_statement, dev_stats = compute_eval_metrics(
                            args, loss,
                            golds, preds, probs, exams,
                            reg_loss, censor_times, adv_loss, dev_stats, 'dev')
    print(log_statement)
    return dev_stats


def eval_model(test_data, models, args):
    '''
        Run model on test data, and return test stats (includes loss

        accuracy, etc)
    '''
    if not isinstance(models, dict):
        models = {'model': models}
    models['model'] = models['model'].to(args.device)

    batch_size = args.batch_size // args.batch_splits
    test_stats = init_metrics_dictionary(modes=['test'])
    data_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ignore_None_collate,
        pin_memory=True,
        drop_last=False)

    loss, golds, preds, probs, exams, reg_loss, censor_times, adv_loss = run_epoch(
        data_loader,
        train_model=False,
        truncate_epoch=False,
        models=models,
        optimizers=None,
        args=args)

    log_statement, test_stats = compute_eval_metrics(
                            args, loss,
                            golds, preds, probs, exams,
                            reg_loss, censor_times, adv_loss, test_stats, 'test')
    print(log_statement)

    return test_stats

def run_epoch(data_loader, train_model, truncate_epoch, models, optimizers, args):
    '''
        Run model for one pass of data_loader, and return epoch statistics.
        args:
        - data_loader: Pytorch dataloader over some dataset.
        - train_model: True to train the model and run the optimizers
        - models: dict of models, where 'model' is the main model, and others can be critics, or meta-models
        - optimizer: dict of optimizers, one for each model
        - args: general runtime args defined in by argparse
        returns:
        - avg_loss: epoch loss
        - golds: labels for all samples in data_loader
        - preds: model predictions for all samples in data_loader
        - probs: model softmaxes for all samples in data_loader
        - exams: exam ids for samples if available, used to cluster samples for evaluation.
    '''
    data_iter = data_loader.__iter__()
    preds = []
    probs = []
    censor_times = []
    golds = []
    losses = []
    adv_losses = []
    reg_losses = []
    exams = []

    torch.set_grad_enabled(train_model)
    for name in models:
        if train_model:
            models[name].train()
            if optimizers is not None:
                optimizers[name].zero_grad()
        else:
            models[name].eval()

    batch_loss = 0
    batch_reg_loss = 0
    batch_adv_loss = 0
    num_batches_per_epoch = len(data_loader)

    if truncate_epoch:
        max_batches =  args.max_batches_per_train_epoch if train_model else args.max_batches_per_dev_epoch
        num_batches_per_epoch = min(len(data_loader), (max_batches * args.batch_splits))


    num_steps = 0
    i = 0
    tqdm_bar = tqdm(data_iter, total=num_batches_per_epoch)
    for batch in data_iter:
        if batch is None:
            warnings.warn('Empty batch')
            continue

        x, y, risk_factors, batch = prepare_batch(batch, args)

        step_results = model_step(x, y, risk_factors, batch, models, optimizers, train_model, args)

        loss, reg_loss, batch_preds, batch_probs, batch_golds, batch_exams, _, adv_loss = step_results
        batch_loss += loss.cpu().data.item()
        batch_reg_loss += reg_loss if isinstance(reg_loss, int) or isinstance(reg_loss, float) else  reg_loss.cpu().data.item()
        batch_adv_loss += adv_loss if isinstance(adv_loss, int) or isinstance(adv_loss, float) else  adv_loss.cpu().data.item()
        if train_model:
            if (i + 1) % args.batch_splits == 0:
                optimizers['model'].step()
                optimizers['model'].zero_grad()

        if (i + 1) % args.batch_splits == 0:
            losses.append(batch_loss)
            reg_losses.append(batch_reg_loss)
            adv_losses.append(batch_adv_loss)
            batch_loss = 0
            batch_reg_loss = 0
            batch_adv_loss = 0

        preds.extend(batch_preds)
        probs.extend(batch_probs)
        golds.extend(batch_golds)
        if batch_exams is not None:
            exams.extend(batch_exams)

        if args.survival_analysis_setup:
            batch_censors = batch['time_at_event'].cpu().numpy()
            censor_times.extend(batch_censors)

        i += 1
        num_steps += 1
        tqdm_bar.update()
        if i > num_batches_per_epoch:
            data_iter.__del__()
            break
    # Recluster results by exam
    if args.cluster_exams:
        aggr = 'majority'
        if 'risk' in args.dataset or 'detection' in args.dataset:
            aggr = 'max'
        gold_set, preds, probs, exam_set = cluster_results_by_exam(golds, preds, probs, exams, aggr=aggr)
        if args.survival_analysis_setup:
            exam_to_censor_details = {exam: {} for exam in exams}
            for censor, exam, gold in zip(censor_times, exams, golds):
                exam_to_censor_details[exam][gold] = censor
            censor_time_set = []
            for exam in exam_set:
                censor_time_set.append( exam_to_censor_details[exam][1] if 1 in exam_to_censor_details[exam] else exam_to_censor_details[exam][0])
            censor_times = censor_time_set
        exams = exam_set
        golds = gold_set

    avg_loss = np.mean(losses)
    avg_reg_loss = np.mean(reg_losses)
    avg_adv_loss = np.mean(adv_losses) if len(adv_losses) > 0 else 0

    return avg_loss, golds, preds, probs, exams, avg_reg_loss, censor_times, avg_adv_loss


def get_hiddens(dataset, models, args):
    '''
        Run model for one pass of dataset, and return np array with hiddens for entire dataset.

        args:
        - dataset: Pytorch dataset obj.
        - models: dict of models, where 'model' is the main model, and others can be critics, or meta-models
        - args: general runtime args defined in by argparse

        returns:
        - hiddens: np array of hiddens
        - img_paths: list of image paths, aligned to hiddens
    '''
    assert not args.use_precomputed_hiddens
    batch_size = args.batch_size // args.batch_splits

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ignore_None_collate,
        drop_last=False)

    for name in models:
        models[name].eval()
        if args.cuda:
            models[name] = models[name].to(args.device)

    volatile = True
    hiddens = []
    img_paths = []
    torch.set_grad_enabled(volatile)
    for batch in tqdm(data_loader):

        if batch is None:
            warnings.warn('Empty batch')
            continue
        with torch.no_grad():
            x, y, risk_factors, batch = prepare_batch(batch, args)
            img_paths.extend(batch['path'])
            step_results = model_step(x, y, risk_factors, batch, models, None, False, args)
            _, _, _, _, _, _, batch_hidden, _ = step_results
            batch_hidden = batch_hidden[:,:args.img_only_dim] if args.use_risk_factors else batch_hidden
            hiddens.extend(batch_hidden)

    hiddens = np.array(hiddens)
    assert len(hiddens) == len(img_paths)
    return hiddens, img_paths

def prepare_batch(batch, args):
    x, y = batch['x'], batch['y']
    if args.cuda:
        x, y = x.to(args.device), y.to(args.device)
    for key in batch.keys():
        if args.cuda:
            if 'region_' in key or 'y_' in key or 'device' in key or key == 'y' or '_seq' in key:
                batch[key] = batch[key].to(args.device)
    if args.use_risk_factors:
        risk_factors = [rf.to(args.device) for rf in batch['risk_factors']]
    else:
        risk_factors = None

    return x, y, risk_factors, batch
