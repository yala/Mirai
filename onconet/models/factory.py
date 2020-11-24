import torch
from torch import nn
from onconet.models.inflate import inflate_model
from onconet.models.blocks.factory import get_block
import pdb

MODEL_REGISTRY = {}

STRIPPING_ERR = 'Trying to strip the model although last layer is not FC.'
NO_MODEL_ERR = 'Model {} not in MODEL_REGISTRY! Available models are {} '
NO_OPTIM_ERR = 'Optimizer {} not supported!'
INVALID_NUM_BLOCKS_ERR = 'Invalid block_layout. Must be length 4. Received {}'
INVALID_BLOCK_SPEC_ERR = 'Invalid block specification. Must be length 2 with (block_name, num_repeats). Received {}'
NUM_MATCHING_LAYERS_MESSAGE = 'Loaded pretrained_weights for {} out of {} parameters.'

def RegisterModel(model_name):
    """Registers a configuration."""

    def decorator(f):
        MODEL_REGISTRY[model_name] = f
        return f

    return decorator


def get_model(args):
    return get_model_by_name(args.model_name, True, args)


def get_model_by_name(name, allow_wrap_model, args):
    '''
        Get model from MODEL_REGISTRY based on args.model_name
        args:
        - name: Name of model, must exit in registry
        - allow_wrap_model: whether or not override args.wrap_model and disable model_wrapping.
        - args: run ime args from parsing

        returns:
        - model: an instance of some torch.nn.Module
    '''
    if not name in MODEL_REGISTRY:
        raise Exception(
            NO_MODEL_ERR.format(
                name, MODEL_REGISTRY.keys()))


    model = MODEL_REGISTRY[name](args)
    allow_data_parallel = 'discriminator' not in name and ('mirai_full' not in args.model_name or allow_wrap_model)
    return wrap_model(model, allow_wrap_model, args, allow_data_parallel)

def wrap_model(model, allow_wrap_model, args, allow_data_parallel=True):
    try:
        model._model.args.use_precomputed_hiddens = args.use_precomputed_hiddens
    except:
        pass
    if args.multi_image and not args.model_name in ['mirai_full']:
        model = inflate_model(model)

    if allow_wrap_model and args.wrap_model:
        model._model = strip_model(model._model)
        if args.patch_size[0] > -1:
            img_size = args.patch_size
        else:
            img_size = args.img_size

        if args.multi_image:
            img_size = ( args.num_images, *args.img_size)
        args.hidden_dim = get_output_size(model, img_size, args.num_chan, args.cuda)

        wrapped_model = ModelWrapper(model, args)
    else:
        wrapped_model = model
    if args.state_dict_path is not None:
        load_pretrained_weights(wrapped_model, torch.load(args.state_dict_path))

    if args.num_gpus > 1 and args.data_parallel and not isinstance(wrapped_model, nn.DataParallel) and allow_data_parallel:
        wrapped_model = nn.DataParallel(wrapped_model,
                                    device_ids=range(args.num_gpus))

    return wrapped_model

def load_model(path, args, do_wrap_model = True):
    print('\nLoading model from [%s]...' % path)
    try:
        model = torch.load(path, map_location='cpu')

        if isinstance(model, dict):
            model = model['model']

        if isinstance(model, nn.DataParallel):
            model = model.module.cpu()
        try:
           model.args.use_pred_risk_factors_at_test = args.use_pred_risk_factors_at_test 
        except:
           pass
        try:
            if hasattr(model, '_model'):
                _model = model._model
            else:
                _model = model
            _model.args.use_pred_risk_factors_at_test = args.use_pred_risk_factors_at_test
            _model.args.use_precomputed_hiddens = args.use_precomputed_hiddens
            _model.args.use_pred_risk_factors_if_unk = args.use_pred_risk_factors_if_unk
            _model.args.pred_risk_factors = args.pred_risk_factors
            _model.args.use_spatial_transformer = args.use_spatial_transformer
        except:
           pass
        try:
            args.img_only_dim = model._model.args.img_only_dim
        except:
            pass
        if do_wrap_model:
            model = {'model': wrap_model(model, True, args)}
    except:
        raise Exception(
            "Sorry, snapshot {} does not exist!".format(path))
    return model

def validate_block_layout(block_layout):
    """Confirms that a block layout is in the right format.

    Arguments:
        block_layout(list): A length n list where each of the n elements
         is a list of lists where each inner list is of length 2 and
         contains (block_name, num_repeats). This specifies the blocks
         in each of the n layers of the ResNet.

    Raises:
        Exception if the block layout is formatted incorrectly.
    """

    # Confirm that each layer is a list of block specifications where
    # each block specification has length 2 (i.e. (block_name, num_repeats))
    for layer_layout in block_layout:
        for block_spec in layer_layout:
            if len(block_spec) != 2:
                raise Exception(INVALID_BLOCK_SPEC_ERR.format(block_spec))


def get_layers(block_layout):
    """Gets the layers for a ResNet given the desired layout of blocks.

    Arguments:
        block_layout(list): A length n list where each of the n elements
         is a list of lists where each inner list is of length 2 and
         contains (block_name, num_repeats). This specifies the blocks
         in each of the n layers of the ResNet.

    Returns:
        layers(list): A list of list of block types conforming to num blocks.
    """

    validate_block_layout(block_layout)

    layers = []
    for layer_layout in block_layout:
        layer = []

        for block_name, num_repeats in layer_layout:
            block = get_block(block_name)
            layer.extend([block]*num_repeats)

        layers.append(layer)

    return layers


def get_params(model):
    '''
    Helper function to get parameters of a model.
    ## TODO: specify parameters to get rather than getting all
    '''

    return model.parameters()


def get_optimizer(model, args):
    '''
    Helper function to fetch optimizer based on args.
    '''
    params = [param for param in model.parameters() if param.requires_grad]
    if args.optimizer == 'adam':
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adagrad':
        return torch.optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(params,
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum )
    else:
        raise Exception(NO_OPTIM_ERR.format(args.optimizer))


def load_pretrained_weights(model, pretrained_state_dict):
    """Loads pretrained weights into a model (even if not all layers match).

    Arguments:
        model(Model): A PyTorch model.
        pretrained_state_dict(dict): A dictionary mapping layer names
            to pretrained weights.
    """
    model_state_dict = model.state_dict()

    # Filter out pretrained layers not in our model
    matching_pretrained_state_dict = {
        layer_name: weights
        for layer_name, weights in pretrained_state_dict.items()
        if (layer_name in model_state_dict and
            pretrained_state_dict[layer_name].size() == model_state_dict[layer_name].size())
    }

    print(NUM_MATCHING_LAYERS_MESSAGE.format(len(matching_pretrained_state_dict),
                                             len(model_state_dict)))
    # Overwrite weights in existing state dict
    model_state_dict.update(matching_pretrained_state_dict)

    # Load the updated state dict
    model.load_state_dict(model_state_dict)


def strip_model(model, num_layers_strip = 1):
    """
    Remove the last pooling anf fc layers from the model.

    :model: model to strip
    :returns: stripped model
    """
    all_children = list(model.named_children() )

    layers_to_strip = all_children[ -1 * num_layers_strip: ]
    for layer_name, layer in layers_to_strip:

        if not type(layer) in [nn.modules.linear.Linear,
                                nn.modules.conv.Conv1d,
                                ModLinear, ModConv1d]:
            raise STRIPPING_ERR
        model._modules[layer_name] = ModelNOP()

    return model



def get_output_size(model, shape, channels, cuda):
    """
    Get the size of the output of the last layer of the model.

    :model: the model
    :shape: shape of the input image tuple(width, height)
    :channels: amount of channels of input image (int)
    :cuda: wether or not to use GPU
    :returns: the size of the output of the last layer of the model.
    """
    bs = 1

    input = torch.rand(bs, channels, *shape)
    if cuda:
        input = input.cuda()
        model = model.cuda()
    output_feat = model.forward(input)
    n_size = output_feat.data.view(bs, -1).size(1)
    return n_size


class ModelNOP(nn.Module):
    def __init__(self):
        '''
            Placeholder nn module. Returns input.
        '''
        super(ModelNOP, self).__init__()


    def forward(self, x):
        return x

class ModelWrapper(nn.Module):
    def __init__(self, model, args):
        '''
            Given some model, add a linear layer and a softmax to fit it the task defined args.dataset
        '''
        super(ModelWrapper, self).__init__()
        self._model = model
        self.args = args
        self.dropout = nn.Dropout(args.dropout)
        if args.make_fc:
            self.last_hidden = nn.Conv1d(1, args.num_classes, args.hidden_dim )
        else:
            self.last_hidden = nn.Linear(args.hidden_dim, args.num_classes)

    def cuda(self, device=None):
        self._model = self._model.cuda(device)
        self.last_hidden = self.last_hidden.cuda(device)

        return self


    def forward(self, x):
        '''
            param x: a batch of image tensors
            returns logit:  logits over args.num_classes for x
        '''
        hidden = self._model(x)
        hidden = self.dropout(hidden)
        hidden = hidden.view(hidden.size()[0], -1)
        if self.args.make_fc:
            logit = self.last_hidden( hidden.unsqueeze(0).transpose(0,1)).squeeze(-1)
        else:
            logit = self.last_hidden(hidden)

        # TODO: It looks like all wrapped models will not work with the current model_step because the current version
        # of the model_step requires that output of a model to be logit, hidden, activ
        return logit, hidden
