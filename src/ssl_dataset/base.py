from dataclasses import dataclass
import logging

import os
import torchvision
from stable_ssl.data.augmentations import TransformsConfig
from stable_ssl.data import base
from stable_ssl.data.base import Sampler
from torch.utils.data import Dataset
from hydra.utils import get_original_cwd
import time
import copy
from hydra.core.hydra_config import HydraConfig
import numpy as np
from dataclasses import dataclass
from torchvision.transforms import v2
from stable_ssl.data.augmentations import TransformConfig



@dataclass
class DisentangledAugmentation:
    """Configuration for the noise to be added to training data.

    Parameters
    ----------
    corruptions : list[tuple[str, dict]]
        The corruptions/transformations to apply. For example:
        ```
        "CustomGaussianNoise":{"severity":1}
        ```
    """

    corruptions: list[dict] = None

    def __post_init__(self):
        """Initialize the corruptions configuration."""
        if self.corruptions is None:
            self.corruptions = [{}]
            self._corruption = v2.Compose([v2.Identity()])
        else:
            self._corruption = v2.Compose(
                [TransformConfig(**t) for t in self.corruptions]
            )

    def __call__(self, x):
        return self._corruption(x)


class FrozenNoiseDataset(Dataset):
    def __init__(self, dataset, noise_transform=None, transforms=None):
        self.dataset = dataset
        self.noise_transform = noise_transform
        self.transforms = transforms
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx

        # self.generate_figure = False  # just switch this to 1 to generate figures with imagenetc_paper.figures.yaml file
        # if HydraConfig.get().job.config_name == "imagenetc_paper_figures.yaml":
        #     self.generate_figure = True
        # self.count = 0
        # # print(HydraConfig.get().overrides.task)  # ['corruption_type=Contrast', 'data_noise=2', 'augmentation_noise=0', 'org_configs@_global_=brown/run_slurm', '++hardware.seed=100']
        # overrides = [x.split("=")[-1] for x in HydraConfig.get().overrides.task]
        # self.run_time = overrides[0] + '_' + overrides[1] + '_' + time.strftime('%H-%M-%S')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]

        # if self.generate_figure:
        #     x.save(get_original_cwd()+f"/figures/{self.run_time}_{idx}_clean.png", format="PNG")

        if self.noise_transform is not None:
            np_random_state = np.random.get_state()
            np.random.seed(idx)
            x = self.noise_transform(x)
            np.random.set_state(np_random_state)

        # if self.generate_figure:
        #     # x.save(get_original_cwd()+f"/figures/{self.run_time}_{idx}_data_noise.png", format="PNG")
        #     cx = copy.deepcopy(x)
        #     copy_transform = copy.deepcopy(self.transforms)
        #     for view in copy_transform.transforms:
        #         view._transform.transforms = view._transform.transforms[:-2]
        #     cx = copy_transform(cx)
        #     if type(cx) is not list:
        #         cx = [cx]
        #     for i in range(len(cx)):
        #         cx[i].save(get_original_cwd()+f"/figures/{self.run_time}_{idx}_{copy_transform.transforms[i].name}.png", format="PNG")
        #     self.count += 1

        if self.transforms is not None:
            x = self.transforms(x)

        # if self.generate_figure and self.count > 1:
        #     exit(0)

        return x, y


@dataclass
class DatasetConfig(base.DatasetConfig):
    """
    Configuration for the data used for training the model.

    Parameters:
    -----------
    dataset_type : str
        The type of dataset: CorruptedTorchvisionDataset or TorchvisionDataset.
        Use CorruptedTorchvisionDataset to support noise in CIFAR and ImageNet.
    corruptions: CorruptionsConfig, optional
        List of corruptions to apply deterministically to the data.
    """

    dataset_type: str = "TorchvisionDataset"
    corruptions: DisentangledAugmentation = None

    def __post_init__(self):
        logging.info(
                f"Using {self.transforms} for augmentations."
            )
        if self.transforms is None:
            self.transforms = [TransformsConfig("None")]
        else:
            self.transforms = [
                TransformsConfig(name, t)
                for name, t in self.transforms.items()
            ]
        logging.info(
                f"Using {self.corruptions} for data corruptions."
            )
        self.corruptions = DisentangledAugmentation(self.corruptions)

    def get_dataset(self):
        """
        Load a dataset from torchvision.datasets (TorchvisionDataset) or
        corruptedTorchvision.datasets (CorruptedTorchvisionDataset).
        """
        if hasattr(torchvision.datasets, self.name):
            if self.name == "ImageNet":
                dataset = torchvision.datasets.ImageNet(
                    root=self.data_path,
                    split=self.split,
                )
            else:
                dataset = getattr(torchvision.datasets, self.name)(
                    root=self.data_path,
                    train=self.split == "train",
                    download=True,
                )
        else:
            dataset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.path),
            )
        # transforms need to be passed to frozen noise dataset and for that to happen we need to override get_dataset
        dataset = FrozenNoiseDataset(dataset, self.corruptions, transforms=Sampler(self.transforms))
        return dataset


@dataclass
class DataConfig(base.DataConfig):
    """
    Configuration for the data used for training the model.

    Parameters:
    -----------
    dataset_type : str
        The type of dataset: CorruptedTorchvisionDataset, TorchvisionDataset.
        Use CorruptedTorchvisionDataset to support noise in CIFAR and ImageNet.
    """

    dataset_type: str

    def __init__(self, train_on, dataset_type, *args, **datasets):
        assert len(args) == 0
        self.train_on = train_on
        self.dataset_type = dataset_type

        self.datasets = {}
        for name, d in datasets.items():
            self.datasets[name] = DatasetConfig(
                dataset_type=self.dataset_type,
                **d)
