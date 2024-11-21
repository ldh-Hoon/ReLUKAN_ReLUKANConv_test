import torch
import torch.nn as nn
import torch.nn.functional as F

from ReLU_KAN import *

class CNN_ReLUKAN(nn.Module):
    def __init__(self, grid_size: int = 5, k: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.k1 = ReLUKAN([128, 10], grid_size, k)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.flat(x)
        x = self.fc1(x)

        x = self.k1(x)
        x = F.log_softmax(x, dim=1)
        return x
