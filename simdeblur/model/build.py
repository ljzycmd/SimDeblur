import copy
from utils.registry import Registry

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
    return build(cfg, LOSS_REGISTRY)


def build_meta_arch(cfg):
    return build(cfg, META_ARCH_REGISTRY)