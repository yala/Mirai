POOL_REGISTRY = {}

NO_POOL_ERR = 'Pool {} not in POOL_REGISTRY! Available pools are {}'

def RegisterPool(pool_name):
    """Registers a pool."""

    def decorator(f):
        POOL_REGISTRY[pool_name] = f
        return f

    return decorator

def get_pool(pool_name):
    """Get pool from POOL_REGISTRY based on pool_name."""

    if not pool_name in POOL_REGISTRY:
        raise Exception(NO_POOL_ERR.format(
            pool_name, POOL_REGISTRY.keys()))

    pool = POOL_REGISTRY[pool_name]

    return pool
