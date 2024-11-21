import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
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
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x
