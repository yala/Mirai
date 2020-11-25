import argparse
import subprocess
import os
import multiprocessing
import pickle
import csv
import json
import sys
from os.path import dirname, realpath
import random

sys.path.append(dirname(dirname(realpath(__file__))))

from onconet.utils import parsing
from onconet.utils.generic import md5

EXPERIMENT_CRASH_MSG = "ALERT! job:[{}] has crashed! Check logfile at:[{}]"
CONFIG_NOT_FOUND_MSG = "ALERT! {} config {} file does not exist!"
RESULTS_PATH_APPEAR_ERR = 'results_path should not appear in config. It will be determined automatically per job'
SUCESSFUL_SEARCH_STR = "SUCCESS! Grid search results dumped to {}."

RESULT_KEY_STEMS = ['{}_loss', '{}_reg_loss', '{}_accuracy', '{}_auc', '{}_precision', '{}_recall', '{}_f1', '{}_c_index', '{}_decile_recall']
RESULT_KEY_STEMS += ['{}_'+'{}year_auc'.format(i) for i in range(1,10)]

ADDITIONAL_RESULT_KEYS = ['dev_auc', 'dev_tnr_by_threshold',
    'dev_fnr_by_threshold', 'test_fnr_by_threshold', 'test_tnr_by_threshold']

LOG_KEYS = ['results_path', 'model_path', 'log_path']
SORT_KEY = 'dev_loss'

parser = argparse.ArgumentParser(description='OncoNet Grid Search Dispatcher. For use information, see `doc/README.md`')
parser.add_argument("--experiment_config_path", required=True, type=str, help="Path of experiment config")
parser.add_argument('--log_dir', type=str, default="logs", help="path to store logs and detailed job level result files")
parser.add_argument('--result_path', type=str, default="results/grid_search.csv", help="path to store grid_search table. This is preferably on shared storage")
parser.add_argument('--rerun_experiments', action='store_true', default=False, help='whether to rerun experiments with the same result file location')
parser.add_argument('--shuffle_experiment_order', action='store_true', default=False, help='whether to shuffle order of experiments')


def launch_experiment(gpu, flag_string):
    '''
    Launch an experiment and direct logs and results to a unique filepath.
    Alert of something goes wrong.
    :gpu: gpu to run this machine on.
    :flag_string: flags to use for this model run. Will be fed into
    scripts/main.py
    '''
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    log_name = md5(flag_string)
    log_stem = os.path.join(args.log_dir, log_name)
    log_path = '{}.txt'.format(log_stem)
    results_path = "{}.results".format(log_stem)

    experiment_string = "CUDA_VISIBLE_DEVICES={} python -u scripts/main.py {} --results_path {}".format(
        gpu, flag_string, results_path)

    # forward logs to logfile
    if "--resume" in flag_string and not args.rerun_experiments:
        pipe_str = ">>"
    else:
        pipe_str = ">"

    shell_cmd = "{} {} {} 2>&1".format(experiment_string, pipe_str, log_path)
    print("Launched exp: {}".format(shell_cmd))

    if not os.path.exists(results_path) or args.rerun_experiments:
        subprocess.call(shell_cmd, shell=True)

    if not os.path.exists(results_path):
        # running this process failed, alert me
        job_fail_msg = EXPERIMENT_CRASH_MSG.format(experiment_string, log_path)
        print(job_fail_msg)

    return results_path, log_path


def worker(gpu, job_queue, done_queue):
    '''
    Worker thread for each gpu. Consumes all jobs and pushes results to done_queue.
    :gpu - gpu this worker can access.
    :job_queue - queue of available jobs.
    :done_queue - queue where to push results.
    '''
    while not job_queue.empty():
        params = job_queue.get()
        if params is None:
            return
        done_queue.put(
            launch_experiment(gpu, params))

def update_sumary_with_results(result_path, log_path, experiment_axies,  summary):
    assert result_path is not None
    try:
        result_dict = pickle.load(open(result_path, 'rb'))
    except Exception as e:
        print("Experiment failed! Logs are located at: {}".format(log_path))
        return summary


    result_dict['log_path'] = log_path
    # Get results from best epoch and move to top level of results dict
    best_epoch_indx = result_dict['epoch_stats']['best_epoch'] if result_dict['train'] else 0
    present_result_keys = []
    for k in result_keys:
        if ( 'test_stats' in result_dict and k in result_dict['test_stats'] and len(result_dict['test_stats'][k])>0) \
            or ('dev_stats' in result_dict and k in result_dict['dev_stats'] and len(result_dict['dev_stats'][k])>0) \
                or (result_dict['train'] and k in result_dict['epoch_stats'] and len(result_dict['epoch_stats'][k])>0):
            present_result_keys.append(k)
            if 'test' in k:
                result_dict[k] = result_dict['test_stats'][k][0]
            elif 'dev' in k:
                result_dict[k] = result_dict['dev_stats'][k][0]
            else:
                result_dict[k] = result_dict['epoch_stats'][k][best_epoch_indx]

    for k in ADDITIONAL_RESULT_KEYS:
        stats_key = 'test_stats' if 'test' in k else 'dev_stats'
        if stats_key in result_dict and k in result_dict[stats_key]:
            if not k in present_result_keys:
                present_result_keys.append(k)
            result_dict[k] = result_dict[stats_key][k][0]

    summary_columns = experiment_axies + present_result_keys + LOG_KEYS
    for prev_summary in summary:
        if len( set(prev_summary.keys()).union(set(summary_columns))) > len(summary_columns):
            summary_columns = list( set(prev_summary.keys()).union(set(summary_columns)) )
    # Only export keys we want to see in sheet to csv
    summary_dict = {}
    for key in summary_columns:
        if key in result_dict:
            summary_dict[key] = result_dict[key]
        else:
            summary_dict[key] = 'NA'
    summary.append(summary_dict)

    if SORT_KEY in summary[0]:
        summary = sorted(summary, key=lambda k: k[SORT_KEY])


    result_dir = os.path.dirname(args.result_path)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # Write summary to csv
    with open(args.result_path, 'w') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=summary_columns)
        writer.writeheader()
        for experiment in summary:
            writer.writerow(experiment)
    return summary

if __name__ == "__main__":

    args = parser.parse_args()
    if not os.path.exists(args.experiment_config_path):
        print(CONFIG_NOT_FOUND_MSG.format("experiment", args.experiment_config_path))
        sys.exit(1)
    experiment_config = json.load(open(args.experiment_config_path, 'r'))

    if 'results_path' in experiment_config['search_space']:
        print(RESULTS_PATH_APPEAR_ERR)
        sys.exit(1)

    job_list, experiment_axies = parsing.parse_dispatcher_config(experiment_config)
    if args.shuffle_experiment_order:
        random.shuffle(job_list)
    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    for job in job_list:
        job_queue.put(job)
    print("Launching Dispatcher with {} jobs!".format(len(job_list)))
    print()
    for gpu in experiment_config['available_gpus']:
        print("Start gpu worker {}".format(gpu))
        multiprocessing.Process(target=worker, args=(gpu, job_queue, done_queue)).start()
    print()

    summary = []
    result_keys = []
    for mode in ['train','dev','test']:
        result_keys.extend( [k.format(mode) for k in RESULT_KEY_STEMS ])

    for i in range(len(job_list)):
        result_path, log_path = done_queue.get()
        summary = update_sumary_with_results(result_path, log_path, experiment_axies, summary)
        dump_result_string = SUCESSFUL_SEARCH_STR.format(args.result_path)
        print("({}/{}) \t {}".format(i+1, len(job_list), dump_result_string))
