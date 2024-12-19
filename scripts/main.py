import pickle
from os.path import dirname, realpath
import sys
import git
sys.path.append(dirname(dirname(realpath(__file__))))
import torch
import onconet.datasets.factory as dataset_factory
import onconet.models.factory as model_factory
from onconet.learn import train
import onconet.transformers.factory as transformer_factory
import onconet.visualize as visualize
import onconet.utils.parsing as parsing
import warnings
import onconet.learn.state_keeper as state
from onconet.utils.get_dataset_stats import get_dataset_stats
import onconet.utils.stats as stats
import pdb
import csv

# Constants
DATE_FORMAT_STR = "%Y-%m-%d:%H-%M-%S"

if __name__ == '__main__':
    args = parsing.parse_args()
    if args.ignore_warnings:
        warnings.simplefilter('ignore')

    repo = git.Repo(search_parent_directories=True)
    commit  = repo.head.object
    args.commit = commit.hexsha
    print(f"OncoNet main running from commit: {commit.hexsha}\n{commit.message}\nauthor: {commit.author}, date: {commit.committed_date}")

    if args.get_dataset_stats:
        print("\nComputing image mean and std...")
        args.img_mean, args.img_std = get_dataset_stats(args)
        print(f'Mean: {args.img_mean}\nStd: {args.img_std}')

    print("\nLoading data-augmentation scheme...")
    transformers = transformer_factory.get_transformers(
        args.image_transformers, args.tensor_transformers, args)
    test_transformers = transformer_factory.get_transformers(
        args.test_image_transformers, args.test_tensor_transformers, args)

    # Load dataset and add dataset specific information to args
    print("\nLoading data...")
    train_data, dev_data, test_data = dataset_factory.get_dataset(args, transformers, test_transformers)

    # Load model and add model specific information to args
    if args.snapshot is None:
        model = model_factory.get_model(args)
    else:
        model = model_factory.load_model(args.snapshot, args)
        if args.replace_snapshot_pool:
            non_trained_model = model_factory.get_model(args)
            model._model.pool = non_trained_model._model.pool
            model._model.args = non_trained_model._model.args

    print(model)

    # Load run parameters if resuming that run.
    args.model_path = state.get_model_path(args)
    print(f'Trained model will be saved to [{args.model_path}]')
    if args.resume:
        try:
            state_keeper = state.StateKeeper(args)
            model, optimizer_state, epoch, lr, epoch_stats = state_keeper.load()
            args.optimizer_state = optimizer_state
            args.current_epoch = epoch
            args.lr = lr
            args.epoch_stats = epoch_stats
        except Exception as e:
            print(f"\nError loading previous state: {e}\nStarting run from scratch.")
    else:
        print("\nRestarting run from scratch.")

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if attr not in ['optimizer_state', 'patient_to_partition_dict', 'path_to_hidden_dict', 'exam_to_year_dict', 'exam_to_device_dict']:
            print(f"\t{attr.upper()}={value}")

    save_path = args.results_path
    print()
    if args.train:
        epoch_stats, model = train.train_model(train_data, dev_data, model, args)
        args.epoch_stats = epoch_stats

        if args.plot_losses:
            visualize.viz_utils.plot_losses(epoch_stats)
        print(f"Save train/dev results to {save_path}")
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path, 'wb'))

    print()
    if args.dev:
        print("-------------\nDev")
        args.dev_stats = train.compute_threshold_and_dev_stats(dev_data, model, args)
        print(f"Save dev results to {save_path}")
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path, 'wb'))

    if args.test:
        print("-------------\nTest")
        args.test_stats = train.eval_model(test_data, model, args)
        print(f"Save test results to {save_path}")
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path, 'wb'))

    if (args.dev or args.test) and args.prediction_save_path is not None:
        exams, probs = [], []
        if args.dev:
            exams.extend(args.dev_stats['exams'])
            probs.extend(args.dev_stats['probs'])
        if args.test:
            exams.extend(args.test_stats['exams'])
            probs.extend(args.test_stats['probs'])
        legend = ['patient_exam_id']
        if args.callibrator_snapshot is not None:
            callibrator = pickle.load(open(args.callibrator_snapshot,'rb'))
        for i in range(args.max_followup):
            legend.append(f"{i+1}_year_risk")
        export = {}
        with open(args.prediction_save_path,'w') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=legend)
            writer.writeheader()
            for exam, arr in zip(exams, probs):
                export['patient_exam_id'] = exam
                for i in range(args.max_followup):
                    key = f"{i+1}_year_risk"
                    raw_val = arr[i]
                    if args.callibrator_snapshot is not None:
                        val = callibrator[i].predict_proba([[raw_val]])[0,1]
                    else:
                        val = raw_val
                    export[key] = val
                writer.writerow(export)
        print(f"Exported predictions to {args.prediction_save_path}")
