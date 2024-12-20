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
    model = MLP(1,[256,256],1)
    model_kan = ReLUKAN([1, 4, 1], 5, 3)
    x = torch.Tensor([np.arange(4, 1024) / 1024]).T
    y = torch.sin(5*torch.pi*x) + torch.log(x)
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
        plt.plot(x.cpu(), y.cpu())
        plt.plot(x.cpu(), pred.cpu())
        plt.plot(x.cpu(), pred_kan.cpu())
        plt.pause(0.001)
        print(loss, loss_kan)