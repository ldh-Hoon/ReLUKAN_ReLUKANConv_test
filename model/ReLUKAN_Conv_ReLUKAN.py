import torch
import torch.nn as nn
import torch.nn.functional as F

from ReLU_KAN_Convolution import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReLUKAN_Conv_ReLUKAN(nn.Module):
    def __init__(self,device=device, grid_size: int = 5, k:int = 3):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            device,
            in_channels=1,
            out_channels= 16,
            kernel_size= (3,3),
            grid_size = grid_size,
            k = k
        )

        self.conv2 = KAN_Convolutional_Layer(
            device,
            in_channels=16,
            out_channels= 32,
            kernel_size = (3,3),
            grid_size = grid_size,
            k = k
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(800, 100)
        self.k1 = ReLUKAN([100, 10], grid_size, k)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)

        x = self.flat(x)

        x = self.fc1(x)
        x = self.k1(x)
        x = F.log_softmax(x, dim=1)
        x = x.squeeze(-1)
        return x