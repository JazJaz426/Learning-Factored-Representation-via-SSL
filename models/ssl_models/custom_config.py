from dataclasses import make_dataclass, dataclass, field
import logging

from omegaconf import OmegaConf

from stable_ssl import Supervised
from stable_ssl.joint_embedding.base import BaseModelConfig
from stable_ssl.joint_embedding.barlow_twins import BarlowTwinsConfig
from stable_ssl.joint_embedding.simclr import SimCLRConfig
from stable_ssl.joint_embedding.vicreg import VICRegConfig
from stable_ssl.joint_embedding.wmse import WMSEConfig
from stable_ssl.config import (
    OptimConfig,
    HardwareConfig,
    LogConfig
)

from .custom_data import DataConfig


_MODEL_CONFIGS = {
    "SimCLR": SimCLRConfig,
    "Barlowtwins": BarlowTwinsConfig,
    "Supervised": BaseModelConfig,
    "VICReg": VICRegConfig,
    "WMSE": WMSEConfig,
}


@dataclass
class TrainerConfig:
    """Global configuration for training a model.

    Parameters
    ----------
    model : BaseModelConfig
        Model configuration.
    data : DataConfig
        Data configuration.
    optim : OptimConfig
        Optimizer configuration.
    hardware : HardwareConfig
        Hardware configuration.
    log : LogConfig
        Logging and checkpointing configuration.
    """

    model: BaseModelConfig = field(default_factory=BaseModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    log: LogConfig = field(default_factory=LogConfig)

    def __repr__(self) -> str:
        """Return a YAML representation of the configuration."""
        return OmegaConf.to_yaml(self)

    def __str__(self) -> str:
        """Return a YAML string of the configuration."""
        return OmegaConf.to_yaml(self)


def get_args(cfg_dict, model_class=None):

    kwargs = {
        name: value
        for name, value in cfg_dict.items()
        if name not in ["data", "optim", "model", "hardware", "log"]
    }

    logging.info(
        f"Using {kwargs.get('corruption_type', None)} corruptions."
    )
    logging.info(
        f"Using {kwargs.get('data_noise', None), kwargs.get('augmentation_noise', None)} for data,augmentation corruptions."
    )

    model = cfg_dict.get("model", {})
    if model_class is None:
        name = model.get("name", None)
    else:
        if issubclass(model_class, Supervised):
            name = "Supervised"
    model = _MODEL_CONFIGS[name](**model)

    cfg_dict.get("data", {})["dataset_type"] = \
        cfg_dict.get("data", {}).get("dataset_type", "TorchvisionDataset")
    args = TrainerConfig(
        model=model,
        data=DataConfig(**cfg_dict.get("data", {})),
        optim=OptimConfig(**cfg_dict.get("optim", {})),
        hardware=HardwareConfig(**cfg_dict.get("hardware", {})),
        log=LogConfig(**cfg_dict.get("log", {})),
    )

    args.__class__ = make_dataclass(
        "TrainerConfig",
        fields=[(name, type(v), v) for name, v in kwargs.items()],
        bases=(type(args),),
    )

    return args
