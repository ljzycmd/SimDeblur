from utils.registry import Registry


DATASET_REGISTRY = Registry("dataset")


def build_dataset(cfg, args=None):
    """
    Build the module with cfg.
    Args:
        cfg (dict): the config of the modules
        registry(Registry): A registry the module belongs to.
    
    Returns:
        The built module.
    """
    args = cfg
    name = args.pop("name")
    dataset = DATASET_REGISTRY.get(name)(args)
    return dataset