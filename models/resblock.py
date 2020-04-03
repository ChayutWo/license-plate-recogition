# Import module and packages
import torch.nn as nn
import torch.nn.functional as F
from utils.variables import *


# Standard CNN ResNet block architecture
class ResBlock(nn.Module):
    """
    A class for a standard Residual Network block

    Structure:
    - conv1: the first 2D convolutional layer
    - bn1: 2D batch normalization
    - conv2: the second 2D convolutional layer
    - bn2: 2D batch normalization

    Method:
    - __init__: initialize a class with n_channels (in_channels and out_channels)
    - forward: perform forward propagation through the residual block and return output
    """

    def __init__(self, n_channels):
        super(ResBlock, self).__init__()
        # Conv1 3x3 stride 1 pad 1 in_channel = out_channel
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(n_channels)
        # Conv2 3x3 stride 1 pad 1 in_channel = out_channel
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        # Perform feed forward propagation through the residual block
        # x -> Conv1 -> bn1 -> ReLU -> Conv2 -> bn2 -> ReLU (with shortcut)
        shortcut = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)) + shortcut)
        return x
