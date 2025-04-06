from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from typing import Optional
from models.learning_head.self_supervised_head import SelfSupervisedCovLearner, SelfSupervisedMaskLearner
from models.learning_head.supervised_head import SupervisedLearner

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)

    def forward(self, x):
        inputs = x
        x = torch.relu(x)
        x = self.conv0(x)
        x = torch.relu(x)
        x = self.conv1(x)
        return x + inputs

class ImpalaCNNLarge(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        backbone_dim: int = 256,
        vector_size_per_factor:int = 3,
        expert_obs: gym.Space= None,
        num_factors: int = None,
        normalized_image: bool = False,
        learning_head:Optional[str]=None
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "ImpalaCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )

        n_input_channels = observation_space.shape[0]
        output_dims = expert_obs.high

        super().__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use ImpalaCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        

        # Modified CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            ResidualBlock(channels= 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            ResidualBlock(channels= 64),
            nn.Flatten()
        )

        # Calculate flattened size
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        self.fc = nn.Linear(n_flatten, backbone_dim)
        

        learning_heads = {'supervised': SupervisedLearner, 'ssl-cov':SelfSupervisedCovLearner, 'ssl-mask':SelfSupervisedMaskLearner}
        self.learning_head = None if learning_head is None or learning_head not in learning_heads else learning_heads[learning_head](backbone_dim=backbone_dim, vector_size_per_factor=vector_size_per_factor, num_factors=output_dims if learning_head == 'supervised' else num_factors)


    def forward(self, x:torch.Tensor, test:bool=True)->torch.Tensor:
        # Pixel normalization (/255)
        x = x / 255.0

        # Forward pass through the Sequential block
        x = self.cnn(x)

        # Pass flattened output through the fully-connected layer
        x = self.fc(x)

        #conditionally apply the learning head on the output from the FC
        if self.learning_head:
            x = self.learning_head(x, test=test)

        return x
    
class ImpalaCNNSmall(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        backbone_dim: int = 256,
        vector_size_per_factor:int = 3,
        expert_obs: gym.Space= None,
        num_factors:int = None,
        normalized_image: bool = False,
        learning_head:Optional[str]=None
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "ImpalaCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)

        output_dims = expert_obs.high
        n_input_channels = observation_space.shape[0]

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use ImpalaCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )

        # Modified CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate flattened size
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        self.fc = nn.Linear(n_flatten, backbone_dim)

        learning_heads = {'supervised': SupervisedLearner, 'ssl-cov':SelfSupervisedCovLearner, 'ssl-mask':SelfSupervisedMaskLearner}
        self.learning_head = None if learning_head is None or learning_head not in learning_heads else learning_heads[learning_head](backbone_dim=backbone_dim, vector_size_per_factor=vector_size_per_factor, num_factors=output_dims if learning_head == 'supervised' else num_factors)



    def forward(self, x:torch.Tensor, test:bool=True)->torch.Tensor:
        # Pixel normalization (/255)
        x = x / 255.0
        
        # Forward pass through the Sequential block
        x = self.cnn(x)

        # Pass flattened output through the fully-connected layer
        x = self.fc(x)

        #conditionally apply the learning head on the output from the FC
        if self.learning_head:
            x = self.learning_head(x, test=test)

        return x