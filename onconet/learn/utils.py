import pdb
import pickle
import json
import warnings
import torch
import numpy as np
from torch.utils import data
import sklearn.metrics
import onconet.utils.stats as stats
from onconet.utils.c_index import concordance_index
from collections import defaultdict

BIRADS_TO_PROB = {'NA':0.0, '1-Negative':0.0, '2-Benign':0.0, '0-Additional imaging needed':1.0, "3-Probably benign": 1.0, "4-Suspicious": 1.0, "5-Highly suspicious": 1.0, "6-Known malignancy": 1.0}
FIRST_UNK_YEAR = 2017

def aggr_maj_vote(preds_of_exam):
    '''
    Take the majority vote of images of the exam.
    '''
    pred_counts = np.bincount(preds_of_exam)
    return np.argmax(pred_counts)

def aggr_max(preds_of_exam):
    '''
    Return the max of the exam predictions
    '''
    return np.ndarray.max( np.array(preds_of_exam), axis=0)

def ignore_None_collate(batch):
    '''
    dataloader.default_collate wrapper that creates batches only of not None values.
    Useful for cases when the dataset.__getitem__ can return None because of some
    exception and then we will want to exclude that sample from the batch.
    '''
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return data.dataloader.default_collate(batch)


def cluster_results_by_exam(golds, preds, probs, exams, aggr='majority'):
    '''
    Aggregate (i.e cluster) gold labels and predictions of images into labels
    and predictions for exams.
    The prediction per exam is decided by the aggr value.
    Args:
    golds : list of gold label for images
    preds : list of predicted labels for images (synced with golds)
    probs : list of output probs for images (synced with golds)
    exams : list of exam identifiers (synced with golds)
    aggr : aggregation function to get the exam prediction from the images [majority, max]
    Returns:
    exam_golds : list of gold lables for exams
    exam_preds : list of predicted labels for exam (synced with exam_golds)
    '''
    if aggr == 'majority':
        aggr_func = aggr_maj_vote
    elif aggr == 'max':
        aggr_func = aggr_max
    else:
        raise('Unknonwn aggr function: {}'.format(aggr))
    preds_by_exam = {}
    probs_by_exam = {}
    golds_by_exam = {}
    for i, exam in enumerate(exams):
        preds_by_exam.setdefault(exam, []).append(preds[i])
        probs_by_exam.setdefault(exam, []).append(probs[i])
        golds_by_exam.setdefault(exam, []).append(golds[i])

    exam_preds = []
    exam_probs = []
    exam_golds = []
    exams = []
    for exam, preds_of_exam in preds_by_exam.items():
        pred = aggr_func(preds_of_exam)
        exam_preds.append(pred)
        prob = aggr_func(probs_by_exam[exam])
        exam_probs.append(prob)
        gold = aggr_func(golds_by_exam[exam])
        exam_golds.append(gold)
        exams.append(exam)

    return exam_golds, exam_preds, exam_probs, exams

def get_human_preds(exams, metadata):
    '''
        get human predictions for a list of exams from the metadata json
    '''
    exam_to_birads = {}
    for row in metadata:
        for exam in row['accessions']:
            exam_to_birads[exam['accession']] = exam['birads']

    human_preds = []
    for exam in exams:
        human_preds.append(BIRADS_TO_PROB[exam_to_birads[exam]])

    return human_preds



def init_metrics_dictionary(modes):
    '''
    Return empty metrics dict
    '''
    stats_dict = defaultdict(list)
    stats_dict['best_epoch'] = 0
    return stats_dict

def get_train_and_dev_dataset_loaders(args, train_data, dev_data, batch_size):
    '''
        Given arg configuration, return appropriate torch.DataLoader
        for train_data and dev_data

        returns:
        train_data_loader: iterator that returns batches
        dev_data_loader: iterator that returns batches
    '''
    if args.class_bal or args.year_weighted_class_bal:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
                weights=train_data.weights,
                num_samples=len(train_data),
                replacement=True)
        train_data_loader = torch.utils.data.DataLoader(
                train_data,
                num_workers=args.num_workers,
                sampler=sampler,
                pin_memory=True,
                batch_size=batch_size,
                collate_fn=ignore_None_collate)
    else:
        train_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=ignore_None_collate,
            pin_memory=True,
            drop_last=True)

    dev_data_loader = torch.utils.data.DataLoader(
        dev_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=ignore_None_collate,
        pin_memory=True,
        drop_last=False)

    return train_data_loader, dev_data_loader


def compute_eval_metrics(args, loss, golds, preds, probs, exams, reg_loss, censor_times, adv_loss, stats_dict, key_prefix):
    stats_dict['{}_loss'.format(key_prefix)].append(loss)
    stats_dict['{}_reg_loss'.format(key_prefix)].append(reg_loss)
    stats_dict['{}_adv_loss'.format(key_prefix)].append(adv_loss)
    stats_dict['preds'] = preds
    stats_dict['golds'] = golds
    stats_dict['censor_times'] = censor_times
    stats_dict['probs'] = probs
    stats_dict['exams'] = exams
    if args.survival_analysis_setup:
        return compute_eval_metrics_survival(args, loss, golds, preds, probs, exams, reg_loss, censor_times, adv_loss, stats_dict, key_prefix)
    else:
        return compute_eval_metrics_classifcation(args, loss, golds, preds, probs, exams, reg_loss, censor_times, adv_loss, stats_dict, key_prefix)

def compute_eval_metrics_survival(args, loss, golds, preds, probs, exams, reg_loss, censor_times, adv_loss, stats_dict, key_prefix):
    log_statement =  '--\n{} - loss: {:.6f} - reg_loss: {:.6f} - adv_loss: {:.6f}'.format(args.objective, loss, reg_loss, adv_loss)
    years = [args.exam_to_year_dict[exam] for exam in exams]

    metrics, sample_sizes = compute_auc_metrics_given_curve(probs, censor_times, golds, years, args.max_followup, args.censoring_distribution)

    for followup in range(args.max_followup):
        for allow_all_years in [True]:
            min_followup_if_neg = followup + 1
            metric_key = min_followup_if_neg if allow_all_years else "year_filtered_{}".format(min_followup_if_neg)
            auc = metrics[metric_key]
            golds_for_eval = sample_sizes[metric_key]
            key_name = '{}_{}year_auc'.format(key_prefix, metric_key)
            log_statement += " -{}: {} (n={} , c={} )".format(key_name, auc, len(golds_for_eval), sum(golds_for_eval))
            stats_dict[key_name].append(auc)

    c_index = metrics['c_index']

    stats_dict['{}_c_index'.format(key_prefix)].append(c_index)
    log_statement += " -c_index: {}".format(c_index)
    stats_dict['{}_decile_recall'.format(key_prefix)].append( metrics['decile_recall'])
    log_statement += " -decile_recall: {}".format(metrics['decile_recall'])

    return log_statement, stats_dict

def compute_auc_metrics_given_curve(probs, censor_times, golds, years, max_followup, censor_distribution):

    metrics = {}
    sample_sizes = {}
    for followup in range(max_followup):
        min_followup_if_neg = followup + 1

        auc, golds_for_eval = compute_auc_x_year_auc(probs, censor_times, golds, followup)
        key = min_followup_if_neg
        metrics[key] = auc
        sample_sizes[key] = golds_for_eval
    try:
        c_index = concordance_index(censor_times, probs, golds, censor_distribution)
    except Exception as e:
            warnings.warn("Failed to calculate C-index because {}".format(e))
            c_index = 'NA'

    metrics['c_index'] = c_index
    end_probs = np.array(probs)[:,-1].tolist()
    sorted_golds = [g for p,g in sorted( zip(end_probs, golds))]
    metrics['decile_recall'] = sum( sorted_golds[-len(sorted_golds)//10:]) / sum(sorted_golds)
    return metrics, sample_sizes


def compute_auc_x_year_auc(probs, censor_times, golds, followup):

    def include_exam_and_determine_label( prob_arr, censor_time, gold):
        valid_pos = gold and censor_time <= followup
        valid_neg = censor_time >= followup
        included, label = (valid_pos or valid_neg), valid_pos
        return included, label

    probs_for_eval, golds_for_eval = [], []
    for prob_arr, censor_time, gold in zip(probs, censor_times, golds):
        include, label = include_exam_and_determine_label(prob_arr, censor_time, gold)
        if include:
            probs_for_eval.append(prob_arr[followup])
            golds_for_eval.append(label)

    try:
        auc = sklearn.metrics.roc_auc_score(golds_for_eval, probs_for_eval, average='samples')
    except Exception as e:
        warnings.warn("Failed to calculate AUC because {}".format(e))
        auc = 'NA'

    return auc, golds_for_eval



def compute_eval_metrics_classifcation(args, loss, golds, preds, probs, exams, reg_loss, censor_times, adv_loss, stats_dict, key_prefix):
    '''
    '''

    accuracy = sklearn.metrics.accuracy_score(y_true=golds, y_pred=preds)
    # Calculate epoch level scores
    if (args.num_classes == 2 or args.predict_birads):
        precision = sklearn.metrics.precision_score(y_true=golds, y_pred=preds)
        recall = sklearn.metrics.recall_score(y_true=golds, y_pred=preds)
        f1 = sklearn.metrics.f1_score(y_true=golds, y_pred=preds)
        try:
            auc = sklearn.metrics.roc_auc_score(golds, probs, average='samples')

        except Exception as e:
            warnings.warn("Failed to calculate AUC because {}".format(e))
            auc = 'NA'
    else:
        auc = 'NA'
        precision = 'NA'
        recall = 'NA'
        f1 = 'NA'

    confusion_matrix = sklearn.metrics.confusion_matrix(golds, preds)
    stats_dict['{}_accuracy'.format(key_prefix)].append(accuracy)
    stats_dict['{}_precision'.format(key_prefix)].append(precision)
    stats_dict['{}_recall'.format(key_prefix)].append(recall)
    stats_dict['{}_f1'.format(key_prefix)].append(f1)
    stats_dict['{}_auc'.format(key_prefix)].append(auc)
    stats_dict['{}_confusion_matrix'.format(key_prefix)].append(confusion_matrix.tolist())

    default_log_statement = '--\n{} - loss: {} reg_loss: {} adv_loss: {} acc: {} auc: {} (n={}, c={}), precision: {} recall: {} f1: {}'.format(
        args.objective, loss, reg_loss, adv_loss, accuracy, auc, len(golds), sum(golds), precision, recall, f1)

    if (args.num_classes == 2 or args.predict_birads) and args.threshold is not None and '1year' in args.dataset:
        try:
            (fnr, fnr_ci), (tpr, tpr_ci), (tnr, tnr_ci) = stats.get_rates_intervals(probs, golds,
                    args.threshold,
                    rebalance_eval_cancers=args.rebalance_eval_cancers)
            stats_dict['{}_fnr_by_threshold'.format(key_prefix)] = [fnr]
            stats_dict['{}_fnr_by_threshold_ci'.format(key_prefix)] = fnr_ci
            stats_dict['{}_tpr_by_threshold'.format(key_prefix)] = [tpr]
            stats_dict['{}_tpr_by_threshold_ci'.format(key_prefix)] = tpr_ci
            stats_dict['{}_tnr_by_threshold'.format(key_prefix)] = [tnr]
            stats_dict['{}_tnr_by_threshold_ci'.format(key_prefix)] = tnr_ci
            stats_dict['threshold'] = args.threshold

            # Compute ROC
            roc_stats = stats.get_roc_stats(golds, probs, key_prefix=key_prefix)
            for k in roc_stats:
                stats_dict[k] = roc_stats[k]

            log_statement += ' FNR: {:.2f} ({:.2f} - {:.2f}), TPR: {:.2f} ({:.2f} - {:.2f}), TNR: {:.2f} ({:.2f} - {:.2f}), Threshold: {:.8f}'.format(
                                            fnr, fnr_ci[0], fnr_ci[1],
                                            tpr, tpr_ci[0], tpr_ci[1],
                                            tnr, tnr_ci[0], tnr_ci[1],
                                            args.threshold
                              )
        except Exception as e:
            warnings.warn("Failed to calculate ROC stats. {}".format(e))
            log_statement = default_log_statement
    else:
        log_statement = default_log_statement

    return log_statement, stats_dict
