from onconet.transformers.basic import ToTensor, ToTensor3d, ToPIL3d, Permute3d
NON_TRANS_ERR = "Transformer {} not in TRANSFORMER_REGISTRY! Available transformers are {}"

IMAGE_TRANSFORMER_REGISTRY = {}
TENSOR_TRANSFORMER_REGISTRY = {}


def RegisterTensorTransformer(name):
    """Registers a dataset."""

    def decorator(obj):
        TENSOR_TRANSFORMER_REGISTRY[name] = obj
        obj.name = name
        return obj

    return decorator


def RegisterImageTransformer(name):
    """Registers a dataset."""

    def decorator(obj):
        IMAGE_TRANSFORMER_REGISTRY[name] = obj
        obj.name = name
        return obj

    return decorator


def get_transformers(image_transformers, tensor_transformers, args):
    transformers = [ToPIL3d()] if args.video else []
    transformers = _add_transformers(transformers, image_transformers,
                                     IMAGE_TRANSFORMER_REGISTRY, args)
    transformers.append(ToTensor3d() if args.video else ToTensor())
    transformers = _add_transformers(transformers, tensor_transformers,
                                     TENSOR_TRANSFORMER_REGISTRY, args)
    if args.video:
        transformers.append(Permute3d())

    return transformers


def _add_transformers(transformers, new_transformers, registry, args):
    for trans in new_transformers:
        name = trans[0]
        kwargs = trans[1]
        if name not in registry:
            raise Exception(NON_TRANS_ERR.format(name, registry.keys()))

        transformers.append(registry[name](args, kwargs))

    return transformers

