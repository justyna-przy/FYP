from typing import List

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,
                              padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)
        return out


class BirdResNet(nn.Module):
    """ResNet‑inspired network for bird classification.

    The network comprises an initial convolution followed by three stages,
    each containing a downsampling block and a configurable number of
    residual blocks.  Finally, global average pooling and a fully connected
    layer produce class scores.

    Args:
        num_classes (int): Number of output classes.
        channels (List[int]): Number of channels for each stage.
        blocks_per_stage (List[int]): Number of residual blocks per stage.
    """

    def __init__(self,
                 num_classes: int,
                 channels: List[int] = [32, 64, 128],
                 blocks_per_stage: List[int] = [2, 3, 3]) -> None:
        super().__init__()

        self.channels = channels

        # Initial 3x3 convolution
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # Build stages
        stages = []
        in_ch = channels[0]
        for stage_idx, (ch, num_blocks) in enumerate(zip(channels, blocks_per_stage)):
            # Downsample block at start of stage except for first stage
            if stage_idx > 0:
                stages.append(DownsampleBlock(in_ch, ch))
            for _ in range(num_blocks):
                stages.append(ResidualBlock(ch))
            in_ch = ch
        self.stages = nn.Sequential(*stages)

        # Final global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu(out)
        out = self.stages(out)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
