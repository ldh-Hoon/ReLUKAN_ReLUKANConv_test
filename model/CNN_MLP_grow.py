import torch
import torch.nn as nn
import torch.nn.functional as F

from ReLU_KAN import *

class CNN_MLP_grow(nn.Module):
    def __init__(self, expand_count=0, dim=128):
        super().__init__()
        self.dim = dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(32 * 7 * 7, dim)

        self.layers = nn.ModuleList([nn.Linear(dim, 10)])

        self.dropout = nn.Dropout(p=0.3)

        self.expand_count = expand_count 

        for _ in range(expand_count):
            new_mlp_layer = nn.Linear(dim, dim)
            self.layers.insert(0, new_mlp_layer)


    def expand_layer(self):
        new_mlp_layer = nn.Linear(self.dim, self.dim)
        self.layers.insert(0, new_mlp_layer)
        self.expand_count += 1

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)

        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)

        x = F.log_softmax(x, dim=1)
        return x

    def check_and_expand(self):
            self.expand_layer()