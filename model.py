from torch import nn

import torch

class GroupedConv3d(nn.Module):
    """3d convolutional block (parallel)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=0),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)) # 3rd channel
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=0),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)) # 3rd channel
        )

    def forward(self, x1, x2):
        x1 = self.conv_block1(x1)
        x2 = self.conv_block2(x2)
        return x1, x2

class GroupedConv2d(nn.Module):
    """2d convolutional block (parallel)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x1, x2):
        x1 = self.conv_block1(x1)
        x2 = self.conv_block2(x2)
        return x1, x2

class GroupedFullyConnected(nn.Module):
    def __init__(self, in_size, out_size, stack=False):
        super().__init__()
        self.stack = stack

        self.ll1 = nn.Linear(in_size, out_size)
        self.ll2 = nn.Linear(in_size, out_size)

    def forward(self, x1, x2):
        self.ll1(x1)
        self.ll2(x2)
        return torch.cat((x1, x2), dim=1) if self.stack else x1, x2

class Flatten(nn.Module):
    def forward(self, x1, x2):
        x1 = torch.flatten(x1, start_dim=1) #? which dim
        x2 = torch.flatten(x2, start_dim=1)
        return x1, x2

class Squeeze(nn.Module):
    def forward(self, x1, x2):
        x1 = torch.squeeze(x1, dim=2) #? which dim
        x2 = torch.squeeze(x2, dim=2)
        return x1, x2

class MonocularVelocityNN(nn.Module):
    def __init__(self, initial_depth):
        super().__init__()

        self.first_conv_block = nn.Sequential(
            GroupedConv3d(in_channels=3, out_channels=96, \
                kernel_size=(3, 11, 11), stride=(3, 4, 4)),

            Squeeze(),

            GroupedConv2d(96, 256, \
                kernel_size=5, stride=1),
            GroupedConv2d(256, 384, \
                kernel_size=3, stride=1),
            GroupedConv2d(384, 384, \
                kernel_size=3, stride=1),

            Flatten()
        )

        self.fully_connected_block = nn.Sequential(
            GroupedFullyConnected(in_size=384, out_size=7680),
            GroupedFullyConnected(in_size=7680, out_size=3840, stack=True)
        )

        self.final_block = nn.Sequential(
            nn.Linear(in_features=7680, out_features=3840),
            nn.Linear(in_features=3840, out_features=3840),
            nn.Linear(in_features=3840, out_features=20) # match output
        )

    def forward(self, x1, x2):
        x1, x2 = self.first_conv_block(x1, x2)
        res = self.fully_connected_block(x1, x2)
        res = self.final_block(x1, x2)
        return res


        