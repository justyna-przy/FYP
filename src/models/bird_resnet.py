"""
ResNet-inspired CNN for bird classification on MAX78002
------------------------------------------------------

This module defines a ResNet‑like convolutional neural network tailored for
classification of mel spectrogram images representing bird calls.  The design
balances accuracy with the memory constraints of the MAX78002 microcontroller.

Architecture overview
~~~~~~~~~~~~~~~~~~~~~

The network follows a simple residual design with three stages.  Each stage
consists of a downsampling block (3×3 convolution with stride 2 followed by
ReLU) and a configurable number of residual blocks.  A residual block
contains two 3×3 convolutions with ReLU activations and a shortcut
connection.  After the final stage, a global average pooling reduces the
feature map to a vector which is passed through a fully connected layer to
produce class scores.

Constraints considered
~~~~~~~~~~~~~~~~~~~~~~

According to the MAX78002 datasheet, the CNN accelerator has approximately
2 MB weight storage and 1.3 MB of data memory.  The model defined here
contains around 1.2 million parameters (in the default configuration) to
stay within the 2 MB budget for 8‑bit weights【387769142224068†L14-L19】.  All
convolutions use 3×3 kernels or 1×1 kernels, and the depth (number of
layers) is modest to avoid exceeding the 128‑layer maximum for MAX78002
【387769142224068†L86-L92】.  Strided convolutions are used instead of
non‑standard operations, and ReLU activations are the only nonlinearity.

Usage
~~~~~

Instantiate the model by passing the number of output classes.  For example,
to create a model for 51 classes (50 bird species plus a non‑bird class):

.. code:: python

    from src.models.bird_resnet import BirdResNet
    model = BirdResNet(num_classes=51)

The input tensor should have shape ``(batch_size, 1, H, W)`` with
values in the range [0, 1].  A typical spectrogram image might be 64×128.

"""

from typing import List

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Basic residual block with two 3×3 convolutions.

    Args:
        channels (int): Number of input and output channels.
    """

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
    """Block that downsamples spatial resolution while increasing channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

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
