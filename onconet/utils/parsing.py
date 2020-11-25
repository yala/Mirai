import argparse
import torch
import os
import pwd
from onconet.datasets.factory import get_dataset_class

EMPTY_NAME_ERR = 'Name of transformer or one of its arguments cant be empty\n\
                  Use "name/arg1=value/arg2=value" format'
BATCH_SIZE_SPLIT_ERR = 'batch_size (={}) should be a multiple of batch_splits (={})'
DATA_AND_MODEL_PARALLEL_ERR = 'data_parallel and model_parallel should not be used in conjunction.'
INVALID_NUM_BLOCKS_ERR = 'Invalid block_layout. Must be length 4. Received {}'
INVALID_BLOCK_SPEC_ERR = 'Invalid block specification. Must be length 2 with "block_name,num_repeats". Received {}'
POSS_VAL_NOT_LIST = 'Flag {} has an invalid list of values: {}. Length of list must be >=1'
CONFLICTING_WEIGHTED_SAMPLING_ERR = 'Cannot both use class_bal and year_weighted_class_bal at the same time.'
INVALID_DATASET_FOR_SURVIVAL = "A dataset with '_full_future'  can only be used with survival_analysis_setup and viceversa."
def parse_transformers(raw_transformers):
    """
    Parse the list of transformers, given by configuration, into a list of
    tuple of the transformers name and a dictionary containing additional args.

    The transformer is assumed to be of the form 'name/arg1=value/arg2=value'

    :raw_transformers: list of strings [unparsed transformers]
    :returns: list of parsed transformers [list of (name,additional_args)]

    """
    transformers = []
    for t in raw_transformers:
        arguments = t.split('/')
        name = arguments[0]
        if name == '':
            raise Exception(EMPTY_NAME_ERR)

        kwargs = {}
        if len(arguments) > 1:
            for a in arguments[1:]:
                splited = a.split('=')
                var = splited[0]
                val = splited[1] if len(splited) > 1 else None
                if var == '':
                    raise Exception(EMPTY_NAME_ERR)

                kwargs[var] = val

        transformers.append((name, kwargs))

    return transformers


def validate_raw_block_layout(raw_block_layout):
    """Confirms that a raw block layout is in the right format.

    Arguments:
        raw_block_layout(list): A list of strings where each string
            is a layer layout in the format
            'block_name,num_repeats-block_name,num_repeats-...'

    Raises:
        Exception if the raw block layout is formatted incorrectly.
    """

    # Confirm that each layer is a list of block specifications where
    # each block specification has length 2 (i.e. block_name,num_repeats)
    for raw_layer_layout in raw_block_layout:
        for raw_block_spec in raw_layer_layout.split('-'):
            if len(raw_block_spec.split(',')) != 2:
                raise Exception(INVALID_BLOCK_SPEC_ERR.format(raw_block_spec))


def parse_block_layout(raw_block_layout):
    """Parses a ResNet block layout, which is a list of layer layouts
    with each layer layout in the form 'block_name,num_repeats-block_name,num_repeats-...'

    Example:
        ['BasicBlock,2',
         'BasicBlock,1-NonLocalBlock,1',
         'BasicBlock,3-NonLocalBlock,2-Bottleneck,2',
         'BasicBlock,2']
        ==>
        [[('BasicBlock', 2)],
         [('BasicBlock', 1), ('NonLocalBlock', 1)],
         [('BasicBlock', 3), ('NonLocalBlock', 2), ('Bottleneck', 2)],
         [('BasicBlock', 2)]]

    Arguments:
        raw_block_layout(list): A list of strings where each string
            is a layer layout as described above.

    Returns:
        A list of lists of length 4 (one for each layer of ResNet). Each inner list is
        a list of tuples, where each tuple is (block_name, num_repeats).
    """

    validate_raw_block_layout(raw_block_layout)

    block_layout = []
    for raw_layer_layout in raw_block_layout:
        raw_block_specs = raw_layer_layout.split('-')
        layer = [raw_block_spec.split(',') for raw_block_spec in raw_block_specs]
        layer = [(block_name, int(num_repeats)) for block_name, num_repeats in layer]
        block_layout.append(layer)

    return block_layout


def parse_dispatcher_config(config):
    '''
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item,
    but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config

    returns: jobs - a list of flag strings, each of which encapsulates one job.
        *Example: --train --cuda --dropout=0.1 ...
    returns: experiment_axies - axies that the grid search is searching over
    '''
    jobs = [""]
    experiment_axies = []
    search_spaces = config['search_space']

    # Support a list of search spaces, convert to length one list for backward compatiblity
    if not isinstance(search_spaces, list):
        search_spaces = [search_spaces]


    for search_space in search_spaces:
        # Go through the tree of possible jobs and enumerate into a list of jobs
        for ind, flag in enumerate(search_space):
            possible_values = search_space[flag]
            if len(possible_values) > 1:
                experiment_axies.append(flag)

            children = []
            if len(possible_values) == 0 or type(possible_values) is not list:
                raise Exception(POSS_VAL_NOT_LIST.format(flag, possible_values))
            for value in possible_values:
                for parent_job in jobs:
                    if type(value) is bool:
                        if value:
                            new_job_str = "{} --{}".format(parent_job, flag)
                        else:
                            new_job_str = parent_job
                    elif type(value) is list:
                        val_list_str = " ".join([str(v) for v in value])
                        new_job_str = "{} --{} {}".format(parent_job, flag,
                                                          val_list_str)
                    else:
                        new_job_str = "{} --{} {}".format(parent_job, flag, value)
                    children.append(new_job_str)
            jobs = children

    return jobs, experiment_axies

def parse_args():
    parser = argparse.ArgumentParser(description='OncoNet Classifier')
    # setup
    parser.add_argument('--run_prefix', default="snapshot", help="what to name this type of model run")
    parser.add_argument('--train', action='store_true', default=False, help='Whether or not to train model')
    parser.add_argument('--test', action='store_true', default=False, help='Whether or not to run model on test set')
    parser.add_argument('--dev', action='store_true', default=False, help='Whether or not to run model on dev set')
    parser.add_argument('--threshold', type=float, default=None, help='Predefine threshold to compute test rates by it')
    parser.add_argument('--ensemble_paths', nargs='*', default=[], help='The list of snapshot paths to build the ensemble from.')
    parser.add_argument('--train_years', nargs='*', type=int, default=[2016,2015,2014, 2013,2012,2011,2010, 2009], help='The list of years to draw training data for mammo risk datasets.')
    parser.add_argument('--dev_years', nargs='*', type=int, default=[2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009], help='The list of years to draw dev data for mammo risk datasets.')
    parser.add_argument('--test_years', nargs='*', type=int, default=[2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009], help='The list of years to draw test data for mammo risk datasets.')
    parser.add_argument('--predict_birads', action='store_true', default=False, help='Wether to predict birads label for negative mammos in risk dataset objects. Note, preds, probs, and labels converted to binary (cancer vs negative) after prediction for logging purposes')
    parser.add_argument('--predict_birads_lambda', type=float, default=0, help='Lambda to weight birads prediction loss')

    parser.add_argument('--invasive_only', action='store_true', default=False, help='Whether or not only consier invasive as pos (and dcis as neg)')
    parser.add_argument('--rebalance_eval_cancers', action='store_true', default=False, help='Wether or not to resample cancers in dev/test to match a the expected cancer incidence.')

    parser.add_argument('--downsample_activ', action='store_true', default=False, help='Whether or not downsample activ')

    # eval
    parser.add_argument('--confidence_interval', type=float, default=0.95, help='Probability mass of test accuracy confidence interval should cover [default: 0.95]')
    parser.add_argument('--num_resamples', type=int, default=10000, help='Number of resamples to use for emperical bootstrap. Should be as large as feasible [default: 1e5]')

    # data
    parser.add_argument('--dataset', default='mnist', help='Name of dataset from dataset factory to use [default: mnist]')
    parser.add_argument('--image_transformers', nargs='*', default=['scale_2d'], help='List of image-transformations to use [default: ["scale_2d"]] \
                        Usage: "--image_transformers trans1/arg1=5/arg2=2 trans2 trans3/arg4=val"')
    parser.add_argument('--tensor_transformers', nargs='*', default=['normalize_2d'], help='List of tensor-transformations to use [default: ["normalize_2d"]]\
                        Usage: similar to image_transformers')
    parser.add_argument('--test_image_transformers', nargs='*', default=['scale_2d'], help='List of image-transformations to use for the dev and test dataset [default: ["scale_2d"]] \
                        Usage: similar to image_transformers')
    parser.add_argument('--test_tensor_transformers', nargs='*', default=['force_num_chan_2d', 'normalize_2d'], help='List of tensor-transformations to use for the dev and test dataset [default: ["force_num_chan_2d", "normalize_2d"]]\
                        Usage: similar to image_transformers')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers for each data loader [default: 4]')
    parser.add_argument('--img_size',  type=int, nargs='+', default=[256, 256], help='width and height of image in pixels. [default: [256,256]')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[-1, -1], help='width and height of patch in pixels. [default: "-1,-1" for no patch mode]')
    parser.add_argument('--get_dataset_stats', action='store_true', default=False, help='Whether to compute the mean and std of the training images on the fly rather than using precomputed values')
    parser.add_argument('--get_activs_instead_of_hiddens',action='store_true', default=False, help='return img feature maps instead of pooled hiddens' )
    parser.add_argument('--img_mean', type=float, nargs='+', default=[0.2023], help='mean value of img pixels. Per channel. ')
    parser.add_argument('--img_std', type=float, nargs='+', default=[0.2576], help='std of img pixels. Per channel. ')
    parser.add_argument('--img_dir', type=str, default='/home/administrator/Mounts/Isilon/pngs16', help='dir of images. Note, image path in dataset jsons should stem from here')
    parser.add_argument('--num_chan', type=int, default=3, help='Number of channels in img. [default:3]')
    parser.add_argument('--force_input_dim', action='store_true', default=False, help='trunctate hiddens from file if not == to input_dim')
    parser.add_argument('--input_dim', type=int, default=512, help='Input dim for 2stage models. [default:512]')
    parser.add_argument('--transfomer_hidden_dim', type=int, default=512, help='start hidden dim for transformer')
    parser.add_argument('--num_heads', type=int, default=8, help='Num heads for transformer')
    parser.add_argument('--multi_image', action='store_true', default=False, help='Whether image will contain multiple slices. Slices could indicate different times, depths, or views')
    parser.add_argument('--num_images', type=int, default=1, help='In multi image setting, the number of images per single sample.')
    parser.add_argument('--pred_both_sides', action='store_true', default=False, help='Simulatenously pred both sides for multi-img model')
    parser.add_argument('--min_num_images', type=int, default=0, help='In multi image setting, the min number of images per single sample.')
    parser.add_argument('--video', action='store_true', default=False, help='Whether the data loaded will be videos (T, H, W, C) or images (H, W, C)')
    parser.add_argument('--metadata_dir', type=str, default=None, help='dir of metadata jsons.')
    parser.add_argument('--metadata_path', type=str, default=None, help='path of metadata csv.')
    parser.add_argument('--cache_path', type=str, default=None, help='dir to cache images.')
    parser.add_argument('--drop_benign_side', action='store_true', default=False, help='If true, drops the samples from a beign breast on an exam that has a malignant one (one datasets that label by side)')

    # sampling
    parser.add_argument('--class_bal', action='store_true', default=False, help='Wether to apply a weighted sampler to balance between the classes on each batch.')
    parser.add_argument('--shift_class_bal_towards_imediate_cancers', action='store_true', default=False, help='Wether to apply a weighted sampler to balance between the classes on each batch.')
    parser.add_argument('--year_weighted_class_bal', action='store_true', default=False, help='Wether to apply a weighted sampler to balance between the classes,year pairs on each batch.')
    parser.add_argument('--device_class_bal', action='store_true', default=False, help='Add device id to dist key for class balancing')
    parser.add_argument('--allowed_devices', nargs='*', default="all", help='List of allowed Devices. None to allow all. Supported are "[Hologic_Selenia, Lorad_Selenia, Selenia_Dimensions, Senograph_DS_ADS_43.10.1]"" ')
    parser.add_argument('--use_c_view_if_available', action='store_true', default=False, help='Wether to to use the C-view if available instead of the regular mammogram.')

    # spatial transformer
    parser.add_argument('--use_spatial_transformer', action='store_true', default=False, help='Wether to add to use a spatial transformer.')
    parser.add_argument('--spatial_transformer_name',  type=str, default='affine', help='Type of transformer to use. Can be affine, bounded_tps or unbounded_tps')
    parser.add_argument('--spatial_transformer_img_size',  nargs='+', default=[208, 256], help='width and height of image in pixels for STN. Must be <= image_size. May downsample to do this')
    parser.add_argument('--location_network_name',  type=str, default='resnet18', help='Model to use for localization network')
    parser.add_argument('--location_network_block_layout', type=str, nargs='+', default=["BasicBlock,2", "BasicBlock,2", "BasicBlock,2", "BasicBlock,2"], help='Layout of blocks for a ResNet model. Must be a list of length 4. Each of the 4 elements is a string of form "block_name,num_repeats-block_name,num_repeats-...". [default: resnet18 layout]')
    parser.add_argument('--tps_grid_size', type=int, default=10, help='Grid size for control points for TPS')
    parser.add_argument('--tps_span_range', type=float, default=0.9, help='Range for relative coords for TPS')
    # regularization
    parser.add_argument('--use_region_annotation', action='store_true', default=False, help='Wether to add a loss factoring in the collected cancer region annotations .')
    parser.add_argument('--fraction_region_annotation_to_use', type=float, default=1.0, help='Fraction of region annotations to use, i.e 1.0 for all and 0 for none. Used for learning curve analysis.')
    parser.add_argument('--region_annotation_loss_type', type=str, default='pred_region', help='Type of region annotation_loss. Options are ["pred_region", "supervised_attention", "constrastive"]')
    parser.add_argument('--region_annotation_pred_kernel_size', type=int, default=5, help='Used for pred-region region annotation loss. Model should use a k*k kernel when deciding if a pixel is in boudning box or not')
    parser.add_argument('--region_annotation_focal_loss_lambda', type=float, default=0, help='Used for pred-region region annotation loss. Lambda to use for focal loss, where 0 corresponds to regular cross entropy.')
    parser.add_argument('--region_annotation_contrast_alpha', type=float, default=.3, help='Used for contrastive region annotation loss. Model should alpha more confident given region')

    parser.add_argument('--regularization_lambda',  type=float, default=0.5,  help='lambda to weigh the region loss.')

    parser.add_argument('--use_adv', action='store_true', default=False, help='Wether to add a adversarial loss representing the kl divergernce from source to target domain. Note, dataset obj must provide "target_x" to take effect.')
    parser.add_argument('--use_mmd_adv', action='store_true', default=False, help='Assume use_adv, use MMD instead of parameterized critic')
    parser.add_argument('--add_repulsive_mmd', action='store_true', default=False, help='add loss to max divergence between representations of + -')
    parser.add_argument('--use_temporal_mmd', action='store_true', default=False, help='Assume use_adv and use_mmd_adv, use Temporal MMD formulation with a discounted LIFO cache')
    parser.add_argument('--temporal_mmd_cache_size', type=int, default=32,  help='max size of cache for mmd.')
    parser.add_argument('--temporal_mmd_discount_factor', type=float, default=.60,  help='decay factor for hidden age. Weight of vector for mmd is discount ** age.')
    parser.add_argument('--adv_loss_lambda',  type=float, default=0.5,  help='lambda to weigh the adversarial loss.')
    parser.add_argument('--train_adv_seperate', action='store_true', default=False, help='Alternate between train adv and train model in model steps, every ')
    parser.add_argument('--anneal_adv_loss',  action='store_true', default=False, help='Assume use_adv and use_mmd_adv, anneal lambda from 0 to set value in 10000 steps')

    parser.add_argument('--turn_off_model_train',  action='store_true', default=False, help='Multiply model loss by 0 so only train adversary.')
    parser.add_argument('--adv_on_logits_alone',  action='store_true', default=False, help='Train adversary using only posterior dist.')
    parser.add_argument('--num_model_steps',  type=int, default=1,  help='num steps of model optimization before switching to adv optimization.')
    parser.add_argument('--num_adv_steps',  type=int, default=100,  help='max num steps of adv before switch back to model optimization. ')
    parser.add_argument('--wrap_model', action='store_true', default=False, help='Whether to strip last layer of model, and add layers to fit to new task.')

    # risk factors
    parser.add_argument('--use_risk_factors', action='store_true', default=False, help='Whether to feed risk factors into last FC of model.') #
    parser.add_argument('--pred_risk_factors', action='store_true', default=False, help='Whether to predict value of all RF from image.') #
    parser.add_argument('--pred_risk_factors_lambda',  type=float, default=0.25,  help='lambda to weigh the risk factor prediction.')
    parser.add_argument('--use_pred_risk_factors_at_test', action='store_true', default=False, help='Whether to use predicted risk factor values at test time.') #
    parser.add_argument('--use_pred_risk_factors_if_unk', action='store_true', default=False, help='Whether to use predicted risk factor values at test time only if rf is unk.') #
    parser.add_argument('--risk_factor_keys', nargs='*', default=['density', 'binary_family_history', 'binary_biopsy_benign', 'binary_biopsy_LCIS', 'binary_biopsy_atypical_hyperplasia', 'age', 'menarche_age', 'menopause_age', 'first_pregnancy_age', 'prior_hist', 'race', 'parous', 'menopausal_status', 'weight','height', 'ovarian_cancer', 'ovarian_cancer_age', 'ashkenazi', 'brca', 'mom_bc_cancer_history', 'm_aunt_bc_cancer_history', 'p_aunt_bc_cancer_history', 'm_grandmother_bc_cancer_history', 'p_grantmother_bc_cancer_history', 'sister_bc_cancer_history', 'mom_oc_cancer_history', 'm_aunt_oc_cancer_history', 'p_aunt_oc_cancer_history', 'm_grandmother_oc_cancer_history', 'p_grantmother_oc_cancer_history', 'sister_oc_cancer_history', 'hrt_type', 'hrt_duration', 'hrt_years_ago_stopped'], help='List of risk factors to include in risk factor vector.')
    parser.add_argument('--risk_factor_metadata_path', type=str, default='/home/administrator/Mounts/Isilon/metadata/risk_factors_jul22_2018_mammo_and_mri.json', help='Path to risk factor metadata file.')
    #survival analysis setup
    parser.add_argument('--survival_analysis_setup', action='store_true', default=False, help='Whether to modify model, eval and training for survival analysis.') #
    parser.add_argument('--make_probs_indep', action='store_true', default=False, help='Make surival model produce indepedent probablities.') #
    parser.add_argument('--mask_mechanism', default='default', help='How to mask for survival objective. options [default, indep, slice, linear].') #
    parser.add_argument('--eval_survival_on_risk', action='store_true', default=False, help='Port over survival model to risk model.') #
    parser.add_argument('--max_followup', type=int, default=5, help='Max followup to predict over')
    parser.add_argument('--eval_risk_survival', action='store_true', default=False, help='Port over risk model to survival model.') #
    # generative modeling
    parser.add_argument('--mask_prob', type=float, default=0, help='Amount of masking to apply on images')
    parser.add_argument('--pred_missing_mammos', action='store_true', default=False, help='Whether to predict missing images when doing image dropout.') #
    parser.add_argument('--also_pred_given_mammos', action='store_true', default=False, help='Whether to predict given images.') #
    parser.add_argument('--pred_missing_mammos_lambda',  type=float, default=0.25,  help='lambda to weigh the pred_missing_mammos.')

    # hiddens based dataset
    parser.add_argument('--use_precomputed_hiddens', action='store_true', default=False, help='Whether to only use hiddens from a pretrained model.')
    parser.add_argument('--zero_out_hiddens',  action='store_true', default=False, help='Zero hiddens from image to test bias')
    parser.add_argument('--use_precomputed_hiddens_in_get_hiddens', action='store_true', default=False, help='Whether to only use hiddens from a pretrained model.')
    parser.add_argument('--hiddens_results_path', type=str, default='/home/administrator/Mounts/Isilon/results/hiddens_from_best_dev_aug_29.results.json', help='Path to results file with hiddens for the whole dataset.')
    parser.add_argument('--use_dev_to_train_model_on_hiddens', action='store_true', default=False, help='Whether to use first half of dev to train model (assuming use_precomputed_hiddens)')
    parser.add_argument('--turn_off_init_projection', action='store_true', default=False, help='Whether to not project base hiddens before use')

    # learning
    parser.add_argument('--optimizer', type=str, default="adam", help='optimizer to use [default: adam]')
    parser.add_argument('--objective', type=str, default="cross_entropy", help='objective function to use [default: cross_entropy]')
    parser.add_argument('--init_lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum to use with SGD')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='initial learning rate [default: 0.5]')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 Regularization penaty [default: 0]')
    parser.add_argument('--patience', type=int, default=10, help='number of epochs without improvement on dev before halving learning rate and reloading best model [default: 5]')
    parser.add_argument('--turn_off_model_reset', action='store_true', default=False, help="Don't reload the model to last best when reducing learning rate")

    parser.add_argument('--tuning_metric', type=str, default='loss', help='Metric to judge dev set results. Possible options include auc, loss, accuracy [default: loss]')
    parser.add_argument('--epochs', type=int, default=256, help='number of epochs for train [default: 256]')
    parser.add_argument('--max_batches_per_train_epoch', type=int, default=10000, help='max batches to per train epoch. [default: 10000]')
    parser.add_argument('--max_batches_per_dev_epoch', type=int, default=10000, help='max batches to per dev epoch. [default: 10000]')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training [default: 128]')
    parser.add_argument('--batch_splits', type=int, default=1, help='Splits batch size into smaller batches in order to fit gpu memmory limits. Optimizer step is run only after one batch size is over. Note: batch_size/batch_splits should be int [default: 1]')
    parser.add_argument('--dropout', type=float, default=0.25, help='Amount of dropout to apply on last hidden layer [default: 0.25]')
    parser.add_argument('--save_dir', type=str, default='snapshot', help='where to dump the model')
    parser.add_argument('--results_path', type=str, default='logs/snapshot', help='where to save the result logs')
    parser.add_argument('--prediction_save_path', type=str, default=None, help='where to save the predictions for dev and test sets')
    parser.add_argument('--no_tuning_on_dev', action='store_true', default=False,  help='Train without tuning on dev (no adaptive lr reduction or saving best model based on dev)')
    parser.add_argument('--lr_reduction_interval', type=int, default=1, help='Number of epochs to wait before reducing lr when training without adaptive lr reduction.')
    parser.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of data to use, i.e 1.0 for all and 0 for none. Used for learning curve analysis.')

    # Alternative training/testing schemes
    parser.add_argument('--ten_fold_cross_val', action='store_true', default=False, help="If true, use 10-fold cross validation.")
    parser.add_argument('--ten_fold_cross_val_seed', type=int, default=1, help="Seed used to generate the partition.")
    parser.add_argument('--ten_fold_test_index', type=int, default=0, help="Index of the partition to hold out as test.")

    # model
    parser.add_argument('--model_name', type=str, default='resnet18', help="Form of model, i.e resnet18, aggregator, revnet, etc.")
    parser.add_argument('--num_layers', type=int, default=3, help="Num layers for transformer based models.")
    parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
    parser.add_argument('--state_dict_path', type=str, default=None, help='filename of model snapshot to load[default: None]')
    parser.add_argument('--img_encoder_snapshot', type=str, default=None, help='filename of img_feat_extractor model snapshot to load. Only used for mirai_full type models [default: None]')
    parser.add_argument('--freeze_image_encoder', action='store_true', default=False, help='Whether freeze image_encoder.') #
    parser.add_argument('--transformer_snapshot', type=str, default=None, help='filename of transformer model snapshot to load. Only used for mirai_full type models [default: None]')
    parser.add_argument('--callibrator_snapshot', type=str, default=None, help='filename of callibrator. Produced for a single model on development set using Platt Scaling')
    parser.add_argument('--patch_snapshot', type=str, default=None, help='filename of patch model snapshot to load. Only used for aggregator type models [default: None]')
    parser.add_argument('--pretrained_on_imagenet', action='store_true', default=False, help='Pretrain the model on imagenet. Only relevant for default models like VGG, resnet etc')
    parser.add_argument('--pretrained_imagenet_model_name', type=str, default='resnet18', help='Name of pretrained model to load for custom resnets.')
    parser.add_argument('--make_fc', action='store_true', default=False, help='Replace last linear layer with convolutional layer')
    parser.add_argument('--replace_bn_with_gn', action='store_true', default=False, help='Use group normalization instead of batch norm.')

    # resnet-specific
    parser.add_argument('--block_layout', type=str, nargs='+', default=["BasicBlock,2", "BasicBlock,2", "BasicBlock,2", "BasicBlock,2"], help='Layout of blocks for a ResNet model. Must be a list of length 4. Each of the 4 elements is a string of form "block_name,num_repeats-block_name,num_repeats-...". [default: resnet18 layout]')
    parser.add_argument('--block_widening_factor', type=int, default=1, help='Factor by which to widen blocks.')
    parser.add_argument('--num_groups', type=int, default=1, help='Num groups per conv in Resnet blocks.')
    parser.add_argument('--pool_name', type=str, default='GlobalAvgPool', help='Pooling mechanism')
    parser.add_argument('--deep_risk_factor_pool', action='store_true', default=False, help='make risk factor pool use several layers to fuse image and rf info')
    parser.add_argument('--replace_snapshot_pool', action='store_true', default=False, help='Use detached models')

    # device
    parser.add_argument('--is_ccds_server', action='store_true', default=False, help='Change all paths accordingly.')
    parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')
    parser.add_argument('--num_gpus', type=int, default=1, help='Num GPUs to use in data_parallel.')
    parser.add_argument('--num_shards', type=int, default=1, help='Num GPUs to shard a single model.')
    parser.add_argument('--data_parallel', action='store_true', default=False, help='spread batch size across all available gpus. Set to false when using model parallelism. The combo of model and data parallelism may result in unexpected behavior')
    parser.add_argument('--model_parallel', action='store_true', default=False, help='spread single model across num_shards. Note must have num_shards > 1 to take effect and only support in specific models. So far supported in all models that extend Resnet-base, i.e resnet-[n], nonlocal-resnet[n], custom-resnet models')

    # visualization
    parser.add_argument('--plot_losses', action='store_true', default=False, help="Produce plots of losses/acc")

    # dataset specific
    parser.add_argument('--cluster_exams', action='store_true', default=False, help='Calculate accuracy of predictions by exam instead of by image')
    parser.add_argument('--background_size', type=int, nargs='+', default=[1024, 1024], help='The size of the background for the mnist_background dataset')
    parser.add_argument('--noise', action='store_true', default=False, help='Whether to add noise to images in the mnist_background dataset')
    parser.add_argument('--noise_var', type=float, default=0.1, help='Variance of gaussian noise added to mnist_background dataset')
    parser.add_argument('--use_permissive_cohort', action='store_true', default=True, help='Allow exams with cancer within 1 year before and after.')
    # GE or Hologic or both types of Mammograms
    parser.add_argument('--mammogram_type', type=str, default=None, help='type of mammograms used to evaluate model')

    # run
    parser.add_argument('--resume', action='store_true', default=False, help='whether to resume if run has already been done')
    parser.add_argument('--ignore_warnings', action='store_true', default=False, help='ignore all warnings')

    args = parser.parse_args()

    # Set args particular to dataset
    get_dataset_class(args).set_args(args)

    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'
    args.unix_username = pwd.getpwuid( os.getuid() )[0]


    # learning initial state
    args.optimizer_state = None
    args.current_epoch = None
    args.lr = None
    args.epoch_stats = None
    args.step_indx = 1

    if args.train_adv_seperate:
        args.max_batches_per_train_epoch *= (args.num_adv_steps+1)

    # Parse list args to appropriate data format
    parse_list_args(args)

    # Check whether certain args or arg combinations are valid
    validate_args(args)

    return args

def validate_args(args):
    """Checks whether certain args or arg combinations are valid.

    Raises:
        Exception if an arg or arg combination is not valid.
    """

    if args.batch_size % args.batch_splits != 0:
        raise ValueError(BATCH_SIZE_SPLIT_ERR.format(args.batch_size, args.batch_splits))

    if args.data_parallel and args.model_parallel:
        raise ValueError(DATA_AND_MODEL_PARALLEL_ERR)

    if args.class_bal and args.year_weighted_class_bal:
        raise ValueError(CONFLICTING_WEIGHTED_SAMPLING_ERR)

    assert args.ten_fold_test_index in range(-1, 10)

def parse_list_args(args):
    """Converts list args to their appropriate data format.

    Includes parsing image dimension args, transformer args,
    block layout args, and more.

    Arguments:
        args(Namespace): Config.

    Returns:
        args but with certain elements modified to be in the
        appropriate data format.
    """

    args.image_transformers = parse_transformers(args.image_transformers)
    args.tensor_transformers = parse_transformers(args.tensor_transformers)
    args.test_image_transformers = parse_transformers(args.test_image_transformers)
    args.test_tensor_transformers = parse_transformers(args.test_tensor_transformers)

    args.block_layout = parse_block_layout(args.block_layout)
