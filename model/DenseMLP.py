import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class DenseMLP(nn.Module):
    def __init__(self, input_shape=784, hidden_shape=64, output_shape=10, num_layers=4):
        super(DenseMLP, self).__init__()

        self.num_layers = num_layers
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_shape = hidden_shape

        self.layers = nn.ModuleList()
        self.weights_past = nn.ParameterList()
        self.weights_future = nn.ParameterList()

        self.layers.append(nn.Linear(input_shape, hidden_shape))

        for i in range(num_layers):
            self.layers.append(nn.Linear(hidden_shape, hidden_shape))

            for j in range(num_layers):
                if i != j: 
                    weight_past = nn.Parameter(torch.empty(hidden_shape, hidden_shape))
                    nn.init.kaiming_uniform_(weight_past, a=math.sqrt(5))
                    self.weights_past.append(weight_past)

                    weight_future = nn.Parameter(torch.empty(hidden_shape, hidden_shape))
                    nn.init.kaiming_uniform_(weight_future, a=math.sqrt(5))
                    self.weights_future.append(weight_future)

        self.fc_out = nn.Linear(hidden_shape, output_shape)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        outputs = [] 
        
        x = self.layers[0](x)
        x = self.silu(x)  
        outputs.append(x)  

        for i in range(1, self.num_layers):
            past_weighted_sum = sum(
                self.weights_past[j * (self.num_layers - 1) + (i - 1)] @ outputs[j].T
                for j in range(i)
            ).T

            x = self.layers[i](past_weighted_sum)
            x = self.silu(x)
            outputs.append(x)

            future_temp_outputs = []
            for j in range(i + 1, self.num_layers):
                future_weighted_sum = sum(
                    self.weights_future[k * (self.num_layers - 1) + (j - 1)] @ outputs[k].T
                    for k in range(i + 1)
                ).T

                future_temp_outputs.append(future_weighted_sum)

            if future_temp_outputs:
                future_weighted_sum = sum(future_temp_outputs) 
                x = x + future_weighted_sum 
                outputs[-1] = x.clone() 

        final_output = self.fc_out(outputs[-1]) 
        final_output = F.log_softmax(final_output, dim=1)
        return final_output 

    def silu(self, x):
        return x * torch.sigmoid(x) 
