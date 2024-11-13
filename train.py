"""
This script demonstrates how to train a model using the stable-SSL library.
"""
import sys
import os

# Add the parent folder to sys.path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(parent_dir))


import hydra
from omegaconf import DictConfig

from models.ssl_models.stable_ssl_patches import patch_stable_ssl
from models.ssl_models.custom_config import get_args

from stable_ssl.joint_embedding import SimCLR, BarlowTwins
from stable_ssl import Supervised

model_dict = {
    "BarlowTwins": BarlowTwins,
    "SimCLR": SimCLR,
    "Supervised": Supervised,
}


@hydra.main(config_path="configs/ssl_configs/")
def main(cfg: DictConfig):
    changed = patch_stable_ssl()
    print(f"Applied {len(changed)} patches to stable-ssl!")
    args = get_args(cfg)

    print("--- Arguments ---")
    print(args)

    trainer = model_dict[args.model.name](args)
    trainer()


if __name__ == "__main__":
    main()