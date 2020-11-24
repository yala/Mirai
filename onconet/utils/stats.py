import numpy as np
import tqdm
import sklearn.metrics
import warnings
import pdb

RESAMPLE_FAILED_WARNING = "Resampling distrubution for estimator {} failed because of : {}"
CONFIDENCE_INTERVAL_EXCEPTION = "Cannot calculate confidence interval. Sampled {} times for found {} (target {}) valid samples."

def ci_bounds(mean, deltas, confidence_interval, num_resamples):
    '''
    Returns the lower and upper bounds of the confidence interval.
    mean : the empirical mean
    deltas : deltas from the empirical mean from different samples
    '''
    deltas = np.sort(deltas)
    index_offset = int((1 - confidence_interval)/2. * num_resamples)
    lower_delta, upper_delta = deltas[index_offset], deltas[-index_offset]
    lower_bound, upper_bound = mean - upper_delta, mean - lower_delta
    return lower_bound, upper_bound


def confidence_interval(confidence_interval, num_resamples, emperical_distribution, estimator=np.mean, clusters=None):
    '''
    Estimates confidence interval of the mean of emperical_distribution
    using emperical bootstrap. Method details are available at:

    https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf


    -confidence_interval: Amount of probability mass interval should cover.
    -num_resamples: Amount of trails to use to estimate confidence interval.
    Should as large as computationally feasiable.
    -emperical_distribution: Emperical distributions for which we want to
    estimate the confidence interval of the mean

    if Clusters is not none, perform two stage bootstrap.
    '''

    emp_mean = estimator(emperical_distribution)
    num_samples = len(emperical_distribution)
    deltas = []
    num_tries = 0
    if clusters is not None:
        cluster_to_inds = {}
        for ind, cluster in enumerate(clusters):
            if cluster not in cluster_to_inds:
                cluster_to_inds[cluster] = []
            cluster_to_inds[cluster].append(ind)
        num_clusters = len(cluster_to_inds)
        cluster_ids = list(cluster_to_inds.keys())

    while len(deltas) < num_resamples:
        if num_tries > num_resamples * 10:
            raise Exception(CONFIDENCE_INTERVAL_EXCEPTION.format(num_tries, len(deltas), num_resamples))
        num_tries += 1
        try:
            if clusters is None:
                resample_ind = np.random.choice(a=range(len(emperical_distribution)),
                                        size=num_samples,
                                        replace=True
                                        )
            else:
                resample_clusters = np.random.choice(a=cluster_ids,
                                                    size=num_clusters,
                                                    replace=True)
                resample_ind = []
                for cluster in resample_clusters:
                    resampled_inds_for_cluster = np.random.choice(a=cluster_to_inds[cluster],
                                                                size=len(cluster_to_inds[cluster]),
                                                                replace=True)
                    resample_ind.extend(resampled_inds_for_cluster)

            resample_dist = [ emperical_distribution[ind] for ind in resample_ind]
            delta = estimator(resample_dist) - emp_mean
            deltas.append( delta )
        except Exception as e:
            warnings.warn(RESAMPLE_FAILED_WARNING.format(estimator, e))

    lower_bound, upper_bound = ci_bounds(emp_mean, deltas, confidence_interval, num_resamples)

    return (lower_bound, upper_bound)


def find_threshold(a, amount, side='lower'):
    '''
    finds the threshold that will give a specific amount of values to be "lower" \ "higher or equal" than it.
    '''
    amount = int(round(amount))
    sorted_a = np.sort(a)
    if side == 'lower':
        if a.shape == (0,):
            return 1.0
        return sorted_a[amount]
    elif side == 'upper':
        if a.shape == (0,):
            return 0
        return sorted_a[-amount]
    else:
        raise 'unvalid side: ' + side


def resample_set_by_distribution(sets, probs, length):
    '''
    Returns a new set by sampling from the given sets with repetitions by the given probs.
    sets : list of sets to sample from
    probs : list of probability for each set (should sum to 1)
    length : length of the new built set
    Returns the new set and an np.array with the set index for each element in the new array.
    '''

    probs_array = np.array([])
    concat_array = np.array([])
    set_inds = np.array([])
    for i, cur_set in enumerate(sets):
        cur_probs = np.ones((len(cur_set))) * probs[i] / len(cur_set)
        probs_array = np.append(probs_array, cur_probs)
        concat_array = np.append(concat_array, cur_set)
        set_inds = np.append(set_inds, np.ones(len(cur_set))*i)

    idx = np.random.choice(np.arange(len(concat_array)), size=length, replace=True, p=probs_array)
    return concat_array[idx], set_inds[idx]


def get_rebalanced_cancer_set(probs, golds, human_preds=None, ratio=0.59, rebalance_eval_cancers=False):
    '''
    Returns a new set with the given ratio of cancers if rebalance eval cancers is true.
    probs : list containing the probs
    golds : list of labels (1- cancer, 0 - benign)
    golds : list of human assessments of the exams
    ratio : The ratio of cancer per 100 benign cases.
    0.59 is the ratio according to http://pubs.rsna.org/doi/pdf/10.1148/radiol.2016161174
    rebalance_eval_cancers : whether or not to relabance to ratio. If not, do a normal sample with replacement.
    '''
    probs, golds = np.array(probs), np.array(golds)
    if rebalance_eval_cancers:
        cancer_prob = ratio / 100
        sample_probs = [1 - cancer_prob, cancer_prob]
        cancer_idxs = np.where(golds == 1)[0]
        benign_idxs = np.where(golds == 0)[0]
        resampled_probs, resampled_labels = resample_set_by_distribution(
                                [probs[benign_idxs],
                                    probs[cancer_idxs]],
                                sample_probs, len(probs))
        assert human_preds is None
    else:
        resample_ind = np.random.choice(a=range(len(probs)),
                                        size=len(probs),
                                        replace=True
                                        )
        resampled_probs, resampled_labels = probs[resample_ind], golds[resample_ind]
        if human_preds is not None:
            human_preds = np.array(human_preds)
            resampled_human_preds = human_preds[resample_ind]

    if human_preds is None:
        return resampled_probs, resampled_labels
    else:
        return resampled_probs, resampled_labels, resampled_human_preds

def get_thresholds_interval(probs, golds, human_preds, confidence_interval=.95, num_resamples=1000, rebalance_eval_cancers=False):
    '''
    Returns the mean threshold and the confidence intervals that meet human's FNR.
    Human's FNR is 0.10 according to http://pubs.rsna.org/doi/pdf/10.1148/radiol.2016161174
    probs, golds : lists
    confidence_interval : Amount of probability mass interval should cover.
    num_resamples : Amount of trails to use to estimate confidence interval.
    rebalance_eval_cancers : Whether or not to use
    Returns : Threshold (mean of thresholds), Confidence Interval
    '''

    probs, golds = np.array(probs), np.array(golds)
    thresholds = []
    for _ in tqdm.tqdm(range(num_resamples)):
        resampled_probs, resampled_labels, resampled_human_preds = get_rebalanced_cancer_set(probs, golds, human_preds=human_preds, rebalance_eval_cancers=rebalance_eval_cancers)
        pos_probs = resampled_probs[np.where(resampled_labels == 1)]
        human_negs = resampled_human_preds < .5
        human_pos = resampled_human_preds > .5
        human_tp = human_pos * resampled_labels
        th_tp = np.sort(resampled_probs[np.where(human_tp == 1)])[0]
        allowed_false_amount = 0.08 / 100 * len(probs)
        th_fn = find_threshold(pos_probs, allowed_false_amount)
        th = min(th_tp, th_fn)
        thresholds.append(th)

    th_mean = np.mean(thresholds)
    deltas = np.array(thresholds) - th_mean

    lower_bound, upper_bound = ci_bounds(th_mean, deltas, confidence_interval, num_resamples)

    threshold = np.mean(thresholds)
    return threshold, (lower_bound, upper_bound)


def get_rates_intervals(probs, golds, threshold, confidence_interval=.95, num_resamples=1000, rebalance_eval_cancers=False):
    '''
    probs, golds : lists
    confidence_interval : Amount of probability mass interval should cover.
    num_resamples : Amount of trails to use to estimate confidence interval.
    sample_size : Size of set to create for each iteration (if 0 then len(probs) is used)
    Returns : FNR, TPR, TNR (means) and their Confidence Intervals
    '''
    fnrs = []
    tprs = []
    tnrs = []
    for _ in tqdm.tqdm(range(num_resamples)):
        resampled_probs, resampled_labels = get_rebalanced_cancer_set(probs, golds, rebalance_eval_cancers=rebalance_eval_cancers)
        fnr, tpr, tnr = get_rates_by_threshold(resampled_probs, resampled_labels, threshold)
        fnrs.append(fnr)
        tprs.append(tpr)
        tnrs.append(tnr)

    def mean_and_bounds(a):
        mean = np.mean(a)
        deltas = np.array(a) - mean
        lower_bound, upper_bound = ci_bounds(mean, deltas, confidence_interval, num_resamples)
        return mean, (lower_bound, upper_bound)

    return mean_and_bounds(fnrs), mean_and_bounds(tprs), mean_and_bounds(tnrs)


def get_rates_by_threshold(probs, golds, threshold):
    golds = np.array(golds)
    probs = np.array(probs)

    positives = probs >= threshold
    negatives = probs < threshold

    tp = np.sum(positives * golds)
    fn = np.sum(negatives * golds)
    tn = np.sum(negatives * (1-golds))
    fp = np.sum(positives * (1-golds))

    tpr = 100.0 * tp / (tp + fn)
    tnr = 100.0 * tn / (tn + fp)

    # fpr by the medical defenition (amount of false positives from 100 exams)
    fnr =  100.0 * fn / len(probs)

    return fnr, tpr, tnr


def get_roc_stats(golds, probs, key_prefix='test'):
    '''
        For a set of gold labels and probs of those labels, return
        false/true negative/positive rates, and associated thresholds
    '''
    roc_stats = {}

    fpr, tpr, p_thresholds = sklearn.metrics.roc_curve(golds, probs, pos_label=1)
    roc_stats['{}_fpr'.format(key_prefix)] = fpr
    roc_stats['{}_tpr'.format(key_prefix)] = tpr
    roc_stats['{}_p_thresholds'.format(key_prefix)] = p_thresholds
    probs0 = []
    [probs0.append(1-p) for p in probs]
    fnr, tnr, n_thresholds = sklearn.metrics.roc_curve(golds, probs0, pos_label=0)
    roc_stats['{}_fnr'.format(key_prefix)] = fnr
    roc_stats['{}_tnr'.format(key_prefix)] = tnr
    roc_stats['{}_n_thresholds'.format(key_prefix)] = n_thresholds

    return roc_stats

def harrels_c_index(probs, golds, time_at_event, args):
    '''
    '''

