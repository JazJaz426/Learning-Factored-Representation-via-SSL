"""
We've added this custom_config.py instead of using the config.py of stable-ssl
because there are some datasets which are part of .custom_data that we want to
use. If we're just using the DataConfig of Stable-SSL along with its
transformations then we don't need this custom_config.py file.
"""
import sys
import os

# Add the parent folder to sys.path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(parent_dir))

from dataclasses import make_dataclass, dataclass, field

from omegaconf import OmegaConf

from stable_ssl.supervised import Supervised
from stable_ssl.base import ModelConfig

from stable_ssl.config import (
    OptimConfig,
    HardwareConfig,
    LogConfig,
    WandbConfig,
)

from data.ssl_dataset.base import DataConfig
from models.ssl_models.custom_barlow_twins import BarlowTwinsConfig
from models.ssl_models.factored_models import CovarianceFactorizationConfig, MaskingFactorizationConfig

_MODEL_CONFIGS = {
    "Supervised": ModelConfig,
    "BarlowTwins": BarlowTwinsConfig,
    "CovarianceFactorizationConfig": CovarianceFactorizationConfig,
    "MaskingFactorizationConfig": MaskingFactorizationConfig,
}

_LOG_CONFIGS = {
    "wandb": WandbConfig,
    None: LogConfig,
    "None": LogConfig,
    "json": LogConfig,
    "jsonlines": LogConfig,
}


@dataclass
class GlobalConfig:
    """Global configuration for training a model.

    Parameters
    ----------
    model : ModelConfig
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

    model: ModelConfig = field(default_factory=ModelConfig)
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

    model = cfg_dict.get("model", {})
    if model_class is None:
        name = model.get("name", None)
    else:
        pass
        # if issubclass(model_class, Supervised):
        #     name = "Supervised"
    model = _MODEL_CONFIGS[name](**model)

    # Get the logging API type and configuration.
    log_config = cfg_dict.get("log", {})
    log_api = log_config.get("api", None)
    log = _LOG_CONFIGS[log_api.lower() if log_api else None](**log_config)

    args = GlobalConfig(
        model=model,
        log=log,
        data=DataConfig(**cfg_dict.get("data", {})),
        optim=OptimConfig(**cfg_dict.get("optim", {})),
        hardware=HardwareConfig(**cfg_dict.get("hardware", {})),
    )

    args.__class__ = make_dataclass(
        "GlobalConfig",
        fields=[(name, type(v), v) for name, v in kwargs.items()],
        bases=(type(args),),
    )

    return args
