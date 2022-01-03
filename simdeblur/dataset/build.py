from ..utils.registry import Registry


DATASET_REGISTRY = Registry("dataset")


def build_dataset(cfg):
    """
    Build the module with cfg.
    Args:
        cfg (dict): the config of the modules

    Returns:
        The built module.
    """
    args = cfg
    name = args.get("name")
    dataset = DATASET_REGISTRY.get(name)(args)
    return dataset


def list_datasets(name=None):
    """
    List all available datasets
    Args:
        name: (TODO) list specific losses corresponds to a given name.
    """
    return list(DATASET_REGISTRY._obj_map.keys())