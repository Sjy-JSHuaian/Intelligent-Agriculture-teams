"""分割模型定义与工厂。"""
from .segformer import build_segformer  # noqa: F401
from .unet import build_unet  # noqa: F401
from .vmamba import build_vmamba  # noqa: F401


def build_model(model_cfg: dict):
    """根据配置中的 name 构建对应模型。"""
    cfg = dict(model_cfg)
    name = cfg.pop("name", "unet")
    if name == "unet":
        return build_unet(**cfg)
    if name == "segformer":
        return build_segformer(**cfg)
    if name == "vmamba":
        return build_vmamba(**cfg)
    raise ValueError(f"未知模型: {name}")
