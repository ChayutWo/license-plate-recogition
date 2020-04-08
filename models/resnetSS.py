# Import module and packages
import torch.nn as nn
import torch.nn.functional as F
from models.resblock import ResBlock
from utils.variables import *


# CNN model class
class resnetSS(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        initialize the network: 178,825

        input:
        - in_channels: number of input channels
        - out_channels: number of output channels (softmax)
        """
        super(resnetSS, self).__init__()
        hidden_channels = [8, 16, 16]

        # Conv1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels[0],
                               kernel_size=5, stride=2, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual block 1
        self.res1 = ResBlock(hidden_channels[0])
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Change channel
        self.conv2 = nn.Conv2d(in_channels=hidden_channels[0], out_channels=hidden_channels[1],
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels[1])
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual block 2
        self.res2 = ResBlock(hidden_channels[1])
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual block 3
        self.res3 = ResBlock(hidden_channels[2])

        # Average pool
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # FC1
        self.fc1 = nn.Linear(in_features=960, out_features=out_channels)

    def forward(self, x):
        # x -> conv1 -> bn1 -> ReLU -> maxpool
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        # x -> Resnet block 1
        x = self.res1(x)
        x = self.maxpool2(x)

        # x -> conv2 -> bn2 -> ReLU
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.maxpool3(x)

        # x -> Resnet block 2
        x = self.res2(x)
        x = self.maxpool4(x)

        # x -> Resnet block 3
        x = self.res3(x)

        # Average pool
        x = self.avgpool(x)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # Linear
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        # Return the dimension of x
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
