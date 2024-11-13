"""
This script demonstrates how to train a model using the stable-SSL library.
"""

import hydra
from omegaconf import DictConfig

from ssl_models.stable_ssl_patches import patch_stable_ssl
from ssl_models.custom_config import get_args

from stable_ssl.joint_embedding import SimCLR
from stable_ssl import Supervised

model_dict = {
    "SimCLR": SimCLR,
    "Supervised": Supervised,
}


@hydra.main(config_path="configs/")
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
