import sys
import os

# Add the parent folder to sys.path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(parent_dir))

import logging

import torchvision
from stable_ssl.data.augmentations import TransformsConfig
from stable_ssl.data import base
from stable_ssl.data.base import Sampler
from torch.utils.data import Dataset
from dataclasses import dataclass
from data.dataset import CustomDataset
from typing import Optional


class GridworldDataset(Dataset):
    def __init__(self, dataset, mode=None, transforms=None):
        self.dataset = dataset
        self.mode = mode
        self.transforms = transforms
        self.classes = dataset.classes
        # self.class_to_idx = dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.mode=='seq':
            """
                item_dict["previous_obs"] = obs_pre
                item_dict["current_obs"] = obs_post
                item_dict["previous_state"] = state_pre
                item_dict["current_state"] = state_post
                item_dict["previous_norm_state"] = norm_state_pre
                item_dict["current_norm_state"] = norm_state_post
                item_dict["action"] = action
            """
            item_dict = self.dataset[idx]
            if self.transforms is not None:
                assert len(self.transforms) == 2
                item_dict["previous_obs"] = Sampler(self.transforms[0])(item_dict["previous_obs"])
                item_dict["current_obs"] = Sampler(self.transforms[1])(item_dict["current_obs"])
            x = [item_dict["previous_obs"], item_dict["current_obs"]]
            y = item_dict["previous_norm_state"]
            z = item_dict["action"]
        elif self.mode in ['cont', 'rand']:
            """
                item_dict["previous_obs"] = obs
                item_dict["previous_state"] = state
                item_dict["previous_norm_state"] = norm_state
                item_dict["action"] = action
            """
            item_dict = self.dataset[idx]
            x = item_dict["previous_obs"]
            y = item_dict["previous_norm_state"]
            z = item_dict["action"]
            if self.transforms is not None:
                x = Sampler(self.transforms)(x)  # 1 z gets 2 or more positive views
        else:
            NotImplementedError(f"{self.mode} not implemented since not implemented in the dataset generator.")

        return x, y, z


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

    data_env_config: Optional[str] = None
    limit: Optional[int] = 1000
    policy_model: Optional[str] = None
    model_path: Optional[str] = None
    mode: Optional[str] = 'seq'

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
                    transform=Sampler(self.transforms),
                )
            else:
                dataset = getattr(torchvision.datasets, self.name)(
                    root=self.data_path,
                    train=self.split == "train",
                    download=True,
                    transform=Sampler(self.transforms),
                )
        else:
            dataset = CustomDataset(
                data_env_config = self.data_env_config,
                limit= self.limit,
                policy_model = self.policy_model,
                model_path = self.model_path,
                mode = self.mode,
            )
            # wrapping the dataset to only create a data, label pair from the RL dataset generator.
            dataset = GridworldDataset(
                dataset=dataset,
                mode=self.mode,
                transforms=self.transforms,
            )

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

    def __init__(self, train_on, *args, **datasets):
        assert len(args) == 0
        self.train_on = train_on

        self.datasets = {}
        for name, d in datasets.items():
            self.datasets[name] = DatasetConfig(**d)
