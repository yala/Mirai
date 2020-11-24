SPATIAL_TRANSFORMER_REGISTRY = {}

NO_SPATIAL_TRANSFORMER_ERR = 'Pool {} not in SPATIAL_TRANSFORMER! Available spatial transformers are {}'

def RegisterSpatialTransformer(st_name):
    """Registers a pool."""

    def decorator(f):
        SPATIAL_TRANSFORMER_REGISTRY[st_name] = f
        return f

    return decorator

def get_spatial_transformer(st_name):
    """Get pool from POOL_REGISTRY based on pool_name."""

    if not st_name in SPATIAL_TRANSFORMER_REGISTRY:
        raise Exception(NO_SPATIAL_TRANSFORMER_ERR.format(
            pool_name, SPATIAL_TRANSFORMER_REGISTRY.keys()))

    spatial_transformer = SPATIAL_TRANSFORMER_REGISTRY[st_name]
    return spatial_transformer
