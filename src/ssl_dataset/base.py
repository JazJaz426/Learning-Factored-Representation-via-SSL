import sys
import os

# Add the parent folder to sys.path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(parent_dir))

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
    disentagle : list[tuple[str, dict]]
        The disentaglement/transformations to apply. For example:
        ```
        "AlterOneFactor":{"p":0.5}
        ```
    """

    disentagle: list[dict] = None

    def __post_init__(self):
        """Initialize the corruptions configuration."""
        if self.disentagle is None:
            self.disentagle = [{}]
            self._disentagle = v2.Compose([v2.Identity()])
        else:
            self._disentagle = v2.Compose(
                [TransformConfig(**t) for t in self.disentagle]
            )

    def __call__(self, x):
        return self._disentagle(x)


class FrozenNoiseDataset(Dataset):
    def __init__(self, dataset, disentangle_transform=None, transforms=None):
        self.dataset = dataset
        self.disentangle_transform = disentangle_transform
        self.transforms = transforms
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]

        if self.transforms is not None:
            x = self.transforms(x)  # this will be more than 1 positive view returned

        if self.disentangle_transform is not None:
            x = [self.disentangle_transform(i) for i in x]  # this will need to work on more than one input samples

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
    disentangle: DisentangledAugmentation = None

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
                f"Using {self.disentangle} for compositional disentanglement."
            )
        self.disentangle = DisentangledAugmentation(self.disentangle)

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
        dataset = FrozenNoiseDataset(dataset, self.disentangle, transforms=Sampler(self.transforms))
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
