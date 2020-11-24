## Adapted from: https://raw.githubusercontent.com/CamDavidsonPilon/lifelines/master/lifelines/utils/concordance.py
## Modified to weight by ipcw (inverse probality of censor weight) to fit Uno's C-index
## Modified to use a time-dependent score

import numpy as np
from lifelines.utils.btree import _BTree
from lifelines import KaplanMeierFitter
import pdb

def get_censoring_dist(train_dataset):
    _dataset = train_dataset.dataset
    times, event_observed = [d['time_at_event'] for d in _dataset], [d['y'] for d in _dataset]
    all_observed_times = set(times)
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed)

    censoring_dist = {time: kmf.predict(time) for time in all_observed_times}
    return censoring_dist

def concordance_index(event_times, predicted_scores, event_observed=None, censoring_dist=None):
    """
    Calculates the concordance index (C-index) between two series
    of event times. The first is the real survival times from
    the experimental data, and the other is the predicted survival
    times from a model of some kind.

    The c-index is the average of how often a model says X is greater than Y when, in the observed
    data, X is indeed greater than Y. The c-index also handles how to handle censored values
    (obviously, if Y is censored, it's hard to know if X is truly greater than Y).


    The concordance index is a value between 0 and 1 where:

    - 0.5 is the expected result from random predictions,
    - 1.0 is perfect concordance and,
    - 0.0 is perfect anti-concordance (multiply predictions with -1 to get 1.0)

    Parameters
    ----------
    event_times: iterable
         a length-n iterable of observed survival times.
    predicted_scores: iterable
        a length-n iterable of predicted scores - these could be survival times, or hazards, etc. See https://stats.stackexchange.com/questions/352183/use-median-survival-time-to-calculate-cph-c-statistic/352435#352435
    event_observed: iterable, optional
        a length-n iterable censorship flags, 1 if observed, 0 if not. Default None assumes all observed.

    Returns
    -------
    c-index: float
      a value between 0 and 1.

    References
    -----------
    Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in
    developing models, evaluating assumptions and adequacy, and measuring and
    reducing errors. Statistics in Medicine 1996;15(4):361-87.

    Examples
    --------

    >>> from lifelines.utils import concordance_index
    >>> cph = CoxPHFitter().fit(df, 'T', 'E')
    >>> concordance_index(df['T'], -cph.predict_partial_hazard(df), df['E'])

    """
    event_times = np.asarray(event_times, dtype=float)
    predicted_scores = 1 - np.asarray(predicted_scores, dtype=float)


    if event_observed is None:
        event_observed = np.ones(event_times.shape[0], dtype=float)
    else:
        event_observed = np.asarray(event_observed, dtype=float).ravel()
        if event_observed.shape != event_times.shape:
            raise ValueError("Observed events must be 1-dimensional of same length as event times")

    num_correct, num_tied, num_pairs = _concordance_summary_statistics(event_times, predicted_scores, event_observed, censoring_dist)

    return _concordance_ratio(num_correct, num_tied, num_pairs)


def _concordance_ratio(num_correct, num_tied, num_pairs):
    if num_pairs == 0:
        raise ZeroDivisionError("No admissable pairs in the dataset.")
    return (num_correct + num_tied / 2) / num_pairs


def _concordance_summary_statistics(
    event_times, predicted_event_times, event_observed, censoring_dist
):  # pylint: disable=too-many-locals
    """Find the concordance index in n * log(n) time.

    Assumes the data has been verified by lifelines.utils.concordance_index first.
    """
    # Here's how this works.
    #
    # It would be pretty easy to do if we had no censored data and no ties. There, the basic idea
    # would be to iterate over the cases in order of their true event time (from least to greatest),
    # while keeping track of a pool of *predicted* event times for all cases previously seen (= all
    # cases that we know should be ranked lower than the case we're looking at currently).
    #
    # If the pool has O(log n) insert and O(log n) RANK (i.e., "how many things in the pool have
    # value less than x"), then the following algorithm is n log n:
    #
    # Sort the times and predictions by time, increasing
    # n_pairs, n_correct := 0
    # pool := {}
    # for each prediction p:
    #     n_pairs += len(pool)
    #     n_correct += rank(pool, p)
    #     add p to pool
    #
    # There are three complications: tied ground truth values, tied predictions, and censored
    # observations.
    #
    # - To handle tied true event times, we modify the inner loop to work in *batches* of observations
    # p_1, ..., p_n whose true event times are tied, and then add them all to the pool
    # simultaneously at the end.
    #
    # - To handle tied predictions, which should each count for 0.5, we switch to
    #     n_correct += min_rank(pool, p)
    #     n_tied += count(pool, p)
    #
    # - To handle censored observations, we handle each batch of tied, censored observations just
    # after the batch of observations that died at the same time (since those censored observations
    # are comparable all the observations that died at the same time or previously). However, we do
    # NOT add them to the pool at the end, because they are NOT comparable with any observations
    # that leave the study afterward--whether or not those observations get censored.
    if np.logical_not(event_observed).all():
        return (0, 0, 0)

    observed_times = set(event_times)


    died_mask = event_observed.astype(bool)
    # TODO: is event_times already sorted? That would be nice...
    died_truth = event_times[died_mask]
    ix = np.argsort(died_truth)
    died_truth = died_truth[ix]

    died_pred = predicted_event_times[died_mask][ix]

    censored_truth = event_times[~died_mask]
    ix = np.argsort(censored_truth)
    censored_truth = censored_truth[ix]
    censored_pred = predicted_event_times[~died_mask][ix]

    censored_ix = 0
    died_ix = 0
    times_to_compare = {}
    for time in observed_times:
        times_to_compare[time] = _BTree(np.unique(died_pred[:, int(time)]))
    num_pairs = np.int64(0)
    num_correct = np.int64(0)
    num_tied = np.int64(0)

    # we iterate through cases sorted by exit time:
    # - First, all cases that died at time t0. We add these to the sortedlist of died times.
    # - Then, all cases that were censored at time t0. We DON'T add these since they are NOT
    #   comparable to subsequent elements.
    while True:
        has_more_censored = censored_ix < len(censored_truth)
        has_more_died = died_ix < len(died_truth)
        # Should we look at some censored indices next, or died indices?
        if has_more_censored and (not has_more_died or died_truth[died_ix] > censored_truth[censored_ix]):
            pairs, correct, tied, next_ix, weight = _handle_pairs(censored_truth, censored_pred, censored_ix, times_to_compare, censoring_dist)
            censored_ix = next_ix
        elif has_more_died and (not has_more_censored or died_truth[died_ix] <= censored_truth[censored_ix]):
            pairs, correct, tied, next_ix, weight = _handle_pairs(died_truth, died_pred, died_ix, times_to_compare, censoring_dist)
            for pred in died_pred[died_ix:next_ix]:
                for time in observed_times:
                    times_to_compare[time].insert(pred[int(time)])
            died_ix = next_ix
        else:
            assert not (has_more_died or has_more_censored)
            break

        num_pairs += pairs * weight
        num_correct += correct * weight
        num_tied += tied * weight

    return (num_correct, num_tied, num_pairs)


def _handle_pairs(truth, pred, first_ix, times_to_compare, censoring_dist):
    """
    Handle all pairs that exited at the same time as truth[first_ix].

    Returns
    -------
      (pairs, correct, tied, next_ix)
      new_pairs: The number of new comparisons performed
      new_correct: The number of comparisons correctly predicted
      next_ix: The next index that needs to be handled
    """
    next_ix = first_ix
    truth_time = truth[first_ix]
    weight = 1./(censoring_dist[truth_time]**2)
    while next_ix < len(truth) and truth[next_ix] == truth[first_ix]:
        next_ix += 1
    pairs = len(times_to_compare[truth_time]) * (next_ix - first_ix)
    correct = np.int64(0)
    tied = np.int64(0)
    for i in range(first_ix, next_ix):
        rank, count = times_to_compare[truth_time].rank(pred[i][int(truth_time)])
        correct += rank
        tied += count

    return (pairs, correct, tied, next_ix, weight)
