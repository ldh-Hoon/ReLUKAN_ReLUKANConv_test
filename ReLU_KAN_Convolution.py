#Credits to: https://github.com/detkov/Convolution-From-Scratch/
import torch
import numpy as np
import math
from typing import List, Tuple, Union

from ReLU_KAN import *

def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    batch_size,n_channels,n, m = matrix.shape

    h_out =  np.floor((n + 2 * padding[0] - kernel_side - (kernel_side - 1) * (dilation[0] - 1)) / stride[0]).astype(int) + 1
    w_out = np.floor((m + 2 * padding[1] - kernel_side - (kernel_side - 1) * (dilation[1] - 1)) / stride[1]).astype(int) + 1
    b = [kernel_side // 2, kernel_side// 2]
    return h_out,w_out,batch_size,n_channels

def multiple_convs_kan_conv2d(matrix, #but as torch tensors. Kernel side asume q el kernel es cuadrado
             kernels, 
             kernel_side,
             out_channels,
             stride= (1, 1), 
             dilation= (1, 1), 
             padding= (0, 0),
             device= "cuda"
             ) -> torch.Tensor:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (batch_size, colors, n, m]): 2D matrix to be convolved.
        kernel  (function]): 2D odd-shaped matrix (e.g. 3x3, 5x5, 13x9, etc.).
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """
    h_out, w_out,batch_size,n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    n_convs = len(kernels)
    matrix_out = torch.zeros((batch_size,out_channels,h_out,w_out)).to(device)#estamos asumiendo que no existe la dimension de rgb
    unfold = torch.nn.Unfold((kernel_side,kernel_side), dilation=dilation, padding=padding, stride=stride)
    conv_groups = unfold(matrix[:,:,:,:]).view(batch_size, n_channels,  kernel_side*kernel_side, h_out*w_out).transpose(2, 3)#reshape((batch_size,n_channels,h_out,w_out))
    #for channel in range(n_channels):
    kern_per_out = len(kernels)//out_channels
    #print(len(kernels),out_channels)
    for c_out in range(out_channels):
        out_channel_accum = torch.zeros((batch_size, h_out, w_out), device=device)

        # Aggregate outputs from each kernel assigned to this output channel
        for k_idx in range(kern_per_out):
            kernel = kernels[c_out * kern_per_out + k_idx]
            conv_result = kernel.conv.forward(conv_groups[:, k_idx, :, :].flatten(0, 1))  # Apply kernel with non-linear function
            out_channel_accum += conv_result.view(batch_size, h_out, w_out)

        matrix_out[:, c_out, :, :] = out_channel_accum  # Store results in output tensor
    
    return matrix_out
def add_padding(matrix: np.ndarray, 
                padding: Tuple[int, int]) -> np.ndarray:
    """Adds padding to the matrix. 

    Args:
        matrix (np.ndarray): Matrix that needs to be padded. Type is List[List[float]] casted to np.ndarray.
        padding (Tuple[int, int]): Tuple with number of rows and columns to be padded. With the `(r, c)` padding we addding `r` rows to the top and bottom and `c` columns to the left and to the right of the matrix

    Returns:
        np.ndarray: Padded matrix with shape `n + 2 * r, m + 2 * c`.
    """
    n, m = matrix.shape
    r, c = padding
    
    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r : n + r, c : m + c] = matrix
    
    return padded_matrix

def kan_conv2d(
            matrix: torch.Tensor,  # (batch_size, colors, n, m) 2D matrix to be convolved
            kernels: List,  # List of kernel objects
            kernel_side: int,
            out_channels: int,
            stride: Tuple[int, int] = (1, 1),
            dilation: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            device: str = "cuda"
            ) -> torch.Tensor:
    """Performs 2D convolution with multiple kernels over the input matrix.

    Args:
        matrix (torch.Tensor): Input 2D matrix of shape (batch_size, colors, n, m).
        kernels (list): List of kernel objects with a forward method.
        kernel_side (int): Size of the square kernel.
        out_channels (int): Number of output channels.
        stride (Tuple[int, int], optional): Stride along axes. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Dilation along axes. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Padding along axes. Defaults to (0, 0).

    Returns:
        torch.Tensor: 2D Feature map, i.e., matrix after convolution.
    """
    # Ensure input matrix is on the specified device
    matrix = matrix.to(device)

    h_out, w_out, batch_size, n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    matrix_out = torch.zeros((batch_size, out_channels, h_out, w_out), device=device)  # Ensure output is on the correct device
    unfold = torch.nn.Unfold((kernel_side, kernel_side), dilation=dilation, padding=padding, stride=stride).to(device)  # Move to device

    # Unfold the input matrix
    conv_groups = unfold(matrix).view(batch_size, n_channels, kernel_side * kernel_side, h_out * w_out).transpose(2, 3)

    # Iterate through output channels
    for c_out in range(out_channels):
        out_channel_accum = torch.zeros((batch_size, h_out, w_out), device=device)

        # Apply each kernel assigned to this output channel
        for k_idx in range(len(kernels) // out_channels):
            kernel = kernels[c_out * (len(kernels) // out_channels) + k_idx]
            # Flatten conv_groups and apply kernel
            conv_result = kernel.conv.forward(conv_groups[:, :, :, :].flatten(0, 1).unsqueeze(-1).to(device))
            out_channel_accum += conv_result.view(batch_size, h_out, w_out)

        matrix_out[:, c_out, :, :] = out_channel_accum

    return matrix_out

class KAN_Convolutional_Layer(torch.nn.Module):
    def __init__(
            self,
            device,
            in_channels: int = 1,
            out_channels: int = 1,
            kernel_size: tuple = (2,2),
            stride: tuple = (1,1),
            padding: tuple = (0,0),
            dilation: tuple = (1,1),
            grid_size: int = 5,
            k = 3,
        ):
        super(KAN_Convolutional_Layer, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.grid_size = grid_size
        self.k = k
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.convs = torch.nn.ModuleList()
        self.stride = stride
        self.device = device

        # Create n_convs KAN_Convolution objects
        for _ in range(in_channels*out_channels):
            self.convs.append(
                KAN_Convolution(
                    device,
                    kernel_size= kernel_size,
                    stride = stride,
                    padding=padding,
                    dilation = dilation,
                    grid_size=grid_size,
                    k=k
                )
            )

    def forward(self, x: torch.Tensor):
        # If there are multiple convolutions, apply them all
        #if self.n_convs>1:
        return multiple_convs_kan_conv2d(x, self.convs,self.kernel_size[0],self.out_channels,self.stride,self.dilation,self.padding, self.device)

        # If there is only one convolution, apply it
        #return self.convs[0].forward(x)


class KAN_Convolution(torch.nn.Module):
    def __init__(
            self,
            device,
            kernel_size: tuple = (2,2),
            stride: tuple = (1,1),
            padding: tuple = (0,0),
            dilation: tuple = (1,1),
            grid_size: int = 5,
            k=3
        ):
        """
        Args
        """
        super(KAN_Convolution, self).__init__()
        self.grid_size = grid_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.device = device
        self.conv = ReLUKANLayer(
            math.prod(kernel_size),
            grid_size,
            3,
            1
        )

    def forward(self, x: torch.Tensor):
        return kan_conv2d(x, self.conv,self.kernel_size[0],1,self.stride,self.dilation,self.padding, self.device)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum( layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)