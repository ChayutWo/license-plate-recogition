# Import module and packages
import torch.nn as nn
import torch.nn.functional as F
from utils.variables import *


# Simple model for quick training
# CNN model class
class simple(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        initialize the network: 15,197

        input:
        - in_channels: number of input channels
        - out_channels: number of output channels (softmax)
        """
        super(simple, self).__init__()
        hidden_channels = [16, 16, 16, 16]
        # Conv1 and then maxpool
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels[0],
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Conv2 and then maxpool
        self.conv2 = nn.Conv2d(in_channels=hidden_channels[0], out_channels=hidden_channels[1],
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(hidden_channels[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Conv3
        self.conv3 = nn.Conv2d(in_channels=hidden_channels[1], out_channels=hidden_channels[2],
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(hidden_channels[2])

        # Conv4
        self.conv4 = nn.Conv2d(in_channels=hidden_channels[2], out_channels=hidden_channels[3],
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(hidden_channels[3])

        # FC1
        self.fc1 = nn.Linear(in_features=384, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=out_channels)

    def forward(self, x):
        # x -> conv1 -> bn1 -> ReLU -> maxpool1
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        # x -> conv2 -> bn2 -> ReLU -> maxpool2
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)

        # x -> conv3 -> bn3 -> ReLU
        x = F.leaky_relu(self.bn3(self.conv3(x)))

        # x -> conv4 -> bn4 -> ReLU
        x = F.leaky_relu(self.bn4(self.conv4(x)))

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # Linear to sigmoid
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        # Return the dimension of x
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
