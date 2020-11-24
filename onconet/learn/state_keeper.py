import pickle
import os
import torch
import collections
import hashlib
import copy

from onconet.utils.generic import md5

OPTIMIZER_PATH = '{}_optim.pt'
PARAM_PATH = '{}_param.p'
STATS_PATH = '{}_stats.p'
MODEL_PATH = '{}_model.pt'
ERROR_MSG = "Sorry, {} does not exist!"

EXCLUDED_ARGS = ['start_time', 'model_path', 'optimizer_state', 'current_epoch', 'lr', 'epoch_stats', 'resume', 'patient_to_partition_dict', 'path_to_hidden_dict']

def get_identifier(args):
    """
    Get a key that represents a specific run. The key should be unique for each set of arguments
    specified for a run so that it can uniquely identify that run. The key will be used to save
    the state of the run.
    Args:
        args: a dictionary of the parameters specified for a run
    Returns:
        key : a string identifying that run
    """
    hash_args = copy.deepcopy(vars(args))
    ## Delete args that could be variable even if it is the same run.

    for attr in EXCLUDED_ARGS:
        if attr in hash_args:
            del hash_args[attr]

    ordered_args = collections.OrderedDict(sorted(hash_args.items(), key=lambda t: t[0]))
    parameters_string = ''.join([str(i) for i in ordered_args.values()])
    key = md5(parameters_string)
    return key

def get_model_path(args):
    key = get_identifier(args)
    return MODEL_PATH.format(os.path.join(args.save_dir, key))


class StateKeeper():
    '''Takes care of saving and loading models for resumable training.'''

    def __init__(self, args):
        self.args = args
        self.identifier = get_identifier(args)

    def save(self, models, optimizers, epoch, lr, epoch_stats):
        """
        Save the state of a run to be loaded later in case it will ger resumed.
        Args:
            models: dictionary of torch models used in the run
            optimizers: dictionary of optimizers used, must correspond one to one with models
            epoch: an integer representing the epoch the run is at
            lr: current learning rate of the optimizer
            epoch_stats: current stats for the run
        """
        ## Save dict for epoch and lr.
        param_dict = {}
        param_dict['epoch'] = epoch
        param_dict['lr'] = lr
        identifier = self.identifier
        param_path = PARAM_PATH.format(os.path.join(self.args.save_dir, identifier))
        with open(param_path, 'wb') as param_file:
            pickle.dump(param_dict, param_file)

        ## Save epoch_stats dict.
        stats_path = STATS_PATH.format(os.path.join(self.args.save_dir, identifier))
        with open(stats_path, 'wb') as stats_file:
            pickle.dump(epoch_stats, stats_file)

        ## Save models and corresponding optimizers
        model_paths = []
        for model_name in models:
            # save model
            model = models[model_name]
            model_path = os.path.join(
                            self.args.save_dir,
                            "{}_{}".format(
                                model_name, MODEL_PATH.format(identifier)))

            torch.save(model, model_path)
            # save optimizer
            optimizer = optimizers[model_name]
            optimizer_path = os.path.join(
                                self.args.save_dir,
                                "{}_{}".format(
                                    model_name, OPTIMIZER_PATH.format(identifier)))
            torch.save(optimizer.state_dict(), optimizer_path)

            model_paths.append(model_path)
        return model_paths


    def load(self):
        """
        Loads the state of a run to resume based on the arguments specified.
        Returns:
            models: a dictionary of the torch models to use in the run to resume
            optimizer_states: a dictionary of the optimizer states to use in the run to resume. One is assumed to exist for each model
            epoch: an integer representing the epoch to resume from
            lr: current learning rate to start from
            epoch_stats: current stats for the run
        """
        identifier = self.identifier
        ## Load dict for epoch and lr.
        param_path = PARAM_PATH.format(os.path.join(self.args.save_dir, identifier))
        try:
            with open(param_path, 'rb') as param_file:
                param_dict = pickle.load(param_file)
        except Exception as e:
            print (e.message)

        ## Load epoch_stats dict.
        stats_path = STATS_PATH.format(os.path.join(self.args.save_dir, identifier))
        try:
            with open(stats_path, 'rb') as stats_file:
                epoch_stats = pickle.load(stats_file)
        except Exception as e:
            print (e.message)


        ## Load model and corresponding optimizers.
        models = {}
        optimizer_states = {}

        model_names = ['model']
        if self.args.use_adv:
            if self.args.use_mmd_adv:
                model_names.extend(['pos_adv', 'neg_adv'])
                if self.args.add_repulsive_mmd:
                    model_names.append('repel_adv')
            else:
                model_names.append('adv')

        for model_name in model_names:
            # Load model

            model_path = os.path.join(
                                self.args.save_dir,
                                "{}_{}".format(
                                    model_name, MODEL_PATH.format(identifier)))
            try:
                models[model_name] = torch.load(model_path)
            except:
                raise Exception(
                    ERROR_MSG.format(model_path))
            print("Loading from " + str(model_path))
            # Load optimizer state
            optimizer_path = os.path.join(
                                self.args.save_dir,
                                "{}_{}".format(
                                    model_name, OPTIMIZER_PATH.format(identifier)))
            try:
                optimizer_states[model_name] = torch.load(optimizer_path)
            except:
                raise Exception(
                    ERROR_MSG.format(optimizer_path))

        return models, optimizer_states, param_dict['epoch'], param_dict['lr'], epoch_stats


    def load_optimizer(self, optimizer, state_dict):
        '''
            Given an optimizer and a state_dict, loads the state_dict into
            the optimizer while preserving correct device placement.

            returns: optimizer, with new state_dict

        '''
        # Build mapping from param to device
        param_to_device = {}
        for param_key in state_dict['state']:
            param = state_dict['state'][param_key]
            for attribute_key in param:
                if isinstance(param[attribute_key], int) or isinstance(param[attribute_key], float):
                    continue
                param_to_device["{}_{}".format(param_key, attribute_key)] = param[attribute_key].get_device()

        optimizer.load_state_dict(state_dict)
        if self.args.cuda:
            # Move params to correct gpus. Load_state_dict uses copy.deepcopy which loses device information
            for param_key in optimizer.state_dict()['state']:
                param = optimizer.state_dict()['state'][param_key]
                for attribute_key in param:
                    if isinstance(param[attribute_key], int) or isinstance(param[attribute_key], float):
                        continue
                    optimizer.state_dict()['state'][param_key][attribute_key] = optimizer.state_dict()['state'][param_key][attribute_key].cuda( param_to_device["{}_{}".format(param_key, attribute_key)])

        return optimizer


