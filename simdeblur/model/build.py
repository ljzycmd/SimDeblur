import copy
from ..utils.registry import Registry

BACKBONE_REGISTRY = Registry("backbone")
LOSS_REGISTRY = Registry("loss")
META_ARCH_REGISTRY = Registry("meta_arch")


def build(cfg, registry, args=None):
    """
    Build the module with cfg.
    Args:
        cfg (dict): the config of the modules
        registry(Registry): A registry the module belongs to.

    Returns:
        The built module.
    """
    args = copy.deepcopy(cfg)
    name = args.pop("name")
    ret = registry.get(name)(**args)
    return ret


def build_backbone(cfg):
    return build(cfg, BACKBONE_REGISTRY)


def build_loss(cfg):
    """
    build a loss from the loss config
    """
    return build(cfg, LOSS_REGISTRY)


def build_meta_arch(cfg):
    name = cfg.meta_arch
    ret = META_ARCH_REGISTRY.get(name)(cfg)
    return ret


def list_backbones(name=None):
    """
    List all available backbones.
    Args:
        name: (TODO) list specific models corresponds to a given name.
    """
    return list(BACKBONE_REGISTRY._obj_map.keys())


def list_meta_archs(name=None):
    """
    List all available meta model architectures
    Args:
        name: (TODO) list specific archs corresponds to a given name.
    """
    return list(META_ARCH_REGISTRY._obj_map.keys())


def list_losses(name=None):
    """
    List all available losses
    Args:
        name: (TODO) list specific losses corresponds to a given name.
    """
    return list(LOSS_REGISTRY._obj_map.keys())
