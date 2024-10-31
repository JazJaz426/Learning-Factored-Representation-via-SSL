import hydra
from omegaconf import DictConfig

import 

from stable_ssl.joint_embedding import SimCLR
from stable_ssl import Supervised

model_dict = {
    "SimCLR": SimCLR,
    "Supervised": Supervised,
}


@hydra.main()
def main(cfg: DictConfig):
    args = noisyssl.get_args(cfg)

    print("--- Arguments ---")
    print(args)

    trainer = model_dict[args.model.name](args)
    trainer()


if __name__ == "__main__":
    main()
