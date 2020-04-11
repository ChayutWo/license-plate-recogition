# Import module and packages
import torch.nn as nn
import torch.nn.functional as F
from models.resblock import ResBlock
from utils.variables import *


# CNN model class
class resnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        initialize the network: 178,825

        input:
        - in_channels: number of input channels
        - out_channels: number of output channels (softmax)
        """
        super(resnet, self).__init__()
        hidden_channels = [16, 16, 32, 32]

        # Conv1 with outchannel 8 and then maxpool
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels[0],
                               kernel_size=5, stride=2, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual block 1
        self.res1 = ResBlock(hidden_channels[0])
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Change channel to 16
        self.conv2 = nn.Conv2d(in_channels=hidden_channels[0], out_channels=hidden_channels[1],
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels[1])
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual block 2
        self.res2 = ResBlock(hidden_channels[1])
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Change channel to 32
        self.conv3 = nn.Conv2d(in_channels=hidden_channels[1], out_channels=hidden_channels[2],
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_channels[2])
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual block 3
        self.res3 = ResBlock(hidden_channels[2])

        # Change channel to 64
        self.conv4 = nn.Conv2d(in_channels=hidden_channels[2], out_channels=hidden_channels[3],
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(hidden_channels[3])

        # Residual block 4
        self.res4 = ResBlock(hidden_channels[3])

        # Average pool
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # FC1 and FC2
        self.fc1 = nn.Linear(in_features=512, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=out_channels)

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

        # x -> conv3 -> bn3 -> ReLU
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.maxpool5(x)

        # x -> Resnet block 3
        x = self.res3(x)

        # x -> conv4 -> bn4 -> ReLU
        x = F.leaky_relu(self.bn4(self.conv4(x)))

        # x -> Resnet block 4
        x = self.res4(x)

        # Average pool
        x = self.avgpool(x)

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
