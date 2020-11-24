import pickle
import pdb
from onconet.utils.c_index import get_censoring_dist
NO_DATASET_ERR = "Dataset {} not in DATASET_REGISTRY! Available datasets are {}"
DATASET_REGISTRY = {}


def RegisterDataset(dataset_name):
    """Registers a dataset."""

    def decorator(f):
        DATASET_REGISTRY[dataset_name] = f
        return f

    return decorator


def get_dataset_class(args):
    if args.dataset not in DATASET_REGISTRY:
        raise Exception(
            NO_DATASET_ERR.format(args.dataset, DATASET_REGISTRY.keys()))

    return DATASET_REGISTRY[args.dataset]

def build_path_to_hidden_dict(args):
    res = pickle.load(open(args.hiddens_results_path,'rb'))
    path_to_hidden = {}
    for split in ['train','dev','test']:
        hiddens, paths = res['{}_hiddens'.format(split)]
        for indx, path in enumerate(paths):
            path_to_hidden[path] = hiddens[indx]
    print("Built path to hidden dict with {} paths, of dim: {}".format(len(path_to_hidden), hiddens[0].shape[0]))
    return path_to_hidden, hiddens[0].shape[0]
# Depending on arg, build dataset
def get_dataset(args, transformers, test_transformers):
    dataset_class = get_dataset_class(args)
    if args.ten_fold_cross_val or args.use_precomputed_hiddens:
        args.patient_to_partition_dict = {}
    if args.use_precomputed_hiddens:
        path_to_hidden_dict, args.hidden_dim = build_path_to_hidden_dict(args)
        if args.force_input_dim:
            args.hidden_dim = args.input_dim
            path_to_hidden_dict = (lambda input_dim, path_to_hidden_dict : {k:v[:input_dim] for k,v in path_to_hidden_dict.items()})(args.input_dim, path_to_hidden_dict)
        
        args.precomputed_hidden_dim = args.hidden_dim
        
    args.exam_to_year_dict = {}
    args.exam_to_device_dict = {}

    train = dataset_class(args, transformers, 'train')
    dev = dataset_class(args, test_transformers, 'dev')
    test = dataset_class(args, test_transformers, 'test')

    if args.survival_analysis_setup:
        args.censoring_distribution = get_censoring_dist(train if len(train) > 0 else test)
    if args.use_precomputed_hiddens:
        train.path_to_hidden_dict = path_to_hidden_dict
        dev.path_to_hidden_dict = path_to_hidden_dict
        test.path_to_hidden_dict = path_to_hidden_dict
    return train, dev, test
