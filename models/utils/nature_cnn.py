from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from typing import Optional
from models.learning_head.self_supervised_head import SelfSupervisedCovLearner, SelfSupervisedMaskLearner, SelfSupervisedCovIKLearner, SelfSupervisedMaskReconstrLearner
from models.learning_head.supervised_head import SupervisedLearner

class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        backbone_dim: int = 512,
        vector_size_per_factor:int = None,
        num_actions: int=3,
        expert_obs: gym.Space= None,
        num_factors: int = None,
        normalized_image: bool = None,
        learning_head:Optional[str]=None
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )

        n_input_channels = observation_space.shape[0]
        output_dims = expert_obs.high

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, backbone_dim), nn.ReLU())

        learning_heads = {'supervised': SupervisedLearner, 'ssl-cov':SelfSupervisedCovLearner, 'ssl-cov-ik':SelfSupervisedCovIKLearner, 'ssl-mask':SelfSupervisedMaskLearner, 'ssl-mask-reconstr':SelfSupervisedMaskReconstrLearner}
        self.learning_head = None if learning_head is None or learning_head not in learning_heads else learning_heads[learning_head](backbone_dim=backbone_dim, vector_size_per_factor=vector_size_per_factor, num_factors=output_dims if learning_head == 'supervised' else num_factors, num_actions = num_actions)

    def forward(self, x: th.Tensor, actions:th.Tensor=None, test:bool=True) -> th.Tensor:
        
        # Forward pass through the Sequential block and fully-connected layer
        x = self.cnn(x)
        x = self.linear(x)

        #conditionally apply the learning head on the output from the FC
        if self.learning_head:
            x = self.learning_head(x, actions=actions, test=test)

        return x