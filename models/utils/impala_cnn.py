from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from typing import Optional

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1)->None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ImpalaCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        vector_size_per_factor:int = 3,
        expert_obs: Optional[int]= None,
        normalized_image: bool = False,
        stop_gradient: bool = False
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "ImpalaCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
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

        n_input_channels = observation_space.shape[0]

        # Wrap convolutional layers and residual blocks into a Sequential block
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            ResidualBlock(16, 32),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            ResidualBlock(16, 32),
            nn.Flatten()
        )

        # Fully connected layer
        # Input size: (48, 88)
        # After conv1 (stride=1): (48, 88)
        # After conv2 (stride=2): (24, 44)
        # Final feature map size: 32 channels * 24 * 44
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.fc = nn.Linear(n_flatten, features_dim)
        
        output_dims = expert_obs.high

        if stop_gradient:
            self.expert_scale_proj = nn.Linear(features_dim, len(output_dims)*vector_size_per_factor)
            self.expert_distribution_proj = nn.ModuleList([nn.Linear(vector_size_per_factor, out_dim+1) for out_dim in output_dims])
            self.softmax = nn.Softmax(dim=-1)

            #extra variables to reshape output
            self.num_dims = len(output_dims)
            self.vector_size_per_factor = vector_size_per_factor
        
        self.stop_gradient = stop_gradient

    def forward(self, x:torch.Tensor, test:bool=True)->torch.Tensor:
        # Pixel normalization (/255)
        x = x / 255.0

        # Forward pass through the Sequential block
        x = self.cnn(x)

        # Pass flattened output through the fully-connected layer
        x = self.fc(x)

        #if in eval mode: used for PPO prediction => then detach computation and take argmax
        if self.stop_gradient and test:
            x = self.expert_scale_proj(x)
            x = x.reshape(x.shape[0], self.num_dims, self.vector_size_per_factor)
            x = [torch.argmax(self.softmax(fc_proj(x[:,i,:])).detach(), dim=-1) for i, fc_proj in enuemrate(self.expert_distribution_proj)]
            x = torch.cat(x, dim=0) #NOTE: hopefully a (batch size, expert_dim) tensor
        
        #if in train mode: used for SL training ==> then do not detach computation and do not take argmax
        if self.stop_gradient and not test:
            x = self.expert_scale_proj(x)
            x = x.reshape(x.shape[0], self.num_dims, self.vector_size_per_factor)
            x = [self.softmax(fc_proj(x[:,i,:])).detach() for i, fc_proj in enuemrate(self.expert_distribution_proj)]
            x = torch.cat(x, dim=0)

        return x