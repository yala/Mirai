BLOCK_REGISTRY = {}

NO_BLOCK_ERR = 'Block {} not in BLOCK_REGISTRY! Available blocks are {}'

def RegisterBlock(block_name):
    """Registers a block."""

    def decorator(f):
        BLOCK_REGISTRY[block_name] = f
        return f

    return decorator

def get_block(block_name):
    """Get block from BLOCK_REGISTRY based on block_name."""

    if not block_name in BLOCK_REGISTRY:
        raise Exception(NO_BLOCK_ERR.format(
            block_name, BLOCK_REGISTRY.keys()))

    block = BLOCK_REGISTRY[block_name]

    return block
