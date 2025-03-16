from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        normalized_image: bool = False,
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

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # Pixel normalization (/255)
        x = x / 255.0

        # Forward pass through the Sequential block
        x = self.cnn(x)

        # Pass flattened output through the fully-connected layer
        x = self.fc(x)
        return x