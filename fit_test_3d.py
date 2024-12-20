# 
# original code from : https://github.com/quiqi/relu_kan/blob/main/fitting_example.py 
# impelement version

import matplotlib.pyplot as plt
from ReLU_KAN import *
import torch
import numpy as np

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
    i = 0
    model = MLP(2,[256,256],1)
    model_kan = ReLUKAN([2, 8, 1], 5, 3)
    #x = torch.Tensor([np.arange(4, 1024) / 1024]).T

    x1 = np.arange(4, 1024) / 1024 
    x2 = np.sin(np.arange(4, 1024) / 1024 * 2 * np.pi) 
    x = np.vstack((x1, x2)).T
    x = torch.Tensor(x)

    y = torch.sin(5*torch.pi*x[:, 0]) + x[:, 1]
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
    losses = []

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
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x.cpu()[:, 0], x.cpu()[:, 1], y.cpu(), label='True Function', color='b', s=1, alpha=0.5)
        ax.scatter(x.cpu()[:, 0], x.cpu()[:, 1], pred.cpu(), label='Predicted MLP', color='r', s=0.1, alpha=0.5)
        ax.scatter(x.cpu()[:, 0], x.cpu()[:, 1], pred_kan.cpu(), label='Predicted KAN', color='g', s=0.1, alpha=0.5)

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')
        ax.legend()
        ax.view_init(elev=30., azim=i)
        i += 1
        
        plt.pause(0.001)
        print(loss, loss_kan)
