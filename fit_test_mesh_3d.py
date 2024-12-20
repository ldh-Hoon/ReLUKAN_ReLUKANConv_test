import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np
import torch.nn as nn

from ReLU_KAN import *

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = MLP(2, [256, 256], 1)
    model_kan = ReLUKAN([2, 8, 16, 8, 1], 5, 3)

    x1 = np.linspace(-1, 0.2, 6) 
    x2 = np.linspace(-1, 0.2, 6)
    x1, x2 = np.meshgrid(x1, x2) 

    x_grid = np.vstack((x1.ravel(), x2.ravel())).T
    x = torch.Tensor(x_grid)

    y = 10 - x[:, 0]**2 - x[:, 1]**2
    y = y.view(-1, 1) 

    if torch.cuda.is_available():
        model = model.cuda()
        model_kan = model_kan.cuda()
        x = x.cuda()
        y = y.cuda()
    
    opt = torch.optim.Adam(model.parameters())
    opt_kan = torch.optim.Adam(model_kan.parameters())
    mse = torch.nn.MSELoss()

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for e in range(1000):
        opt.zero_grad()
        opt_kan.zero_grad()

        pred = model(x)
        pred_kan = model_kan(x)

        loss = mse(pred, y)
        loss.backward()
        opt.step()
        pred = pred.detach()

        loss_kan = mse(pred_kan, y)
        loss_kan.backward()
        opt_kan.step()
        pred_kan = pred_kan.detach()

        plt.clf()

        pred = pred.view(x1.shape) 
        pred_kan = pred_kan.view(x1.shape) 
        yy = y.view(x1.shape)

        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('viridis') 

        pred_flat = pred.cpu().numpy().flatten()
        pred_kan_flat = pred_kan.cpu().numpy().flatten()
        yy_flat = yy.cpu().numpy().flatten()

        color_pred = cmap((pred_flat - np.min(pred_flat)) / (np.max(pred_flat) - np.min(pred_flat)))
        color_kan = cmap((pred_kan_flat - np.min(pred_kan_flat)) / (np.max(pred_kan_flat) - np.min(pred_kan_flat)))
        color_true = cmap((yy_flat - np.min(yy_flat)) / (np.max(yy_flat) - np.min(yy_flat)))

        ax.plot_trisurf(x1.flatten(), x2.flatten(), yy_flat, facecolors=color_true, alpha=0.5)
        ax.plot_trisurf(x1.flatten(), x2.flatten(), pred_flat, facecolors=color_pred, alpha=0.3)
        ax.plot_trisurf(x1.flatten(), x2.flatten(), pred_kan_flat, facecolors=color_kan, alpha=0.2)

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')


        plt.pause(0.001)
        print(loss.item(), loss_kan.item())
