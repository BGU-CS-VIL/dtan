"""
Created on Oct  2019

author: ronsha
"""
import torch
import torch.nn as nn
from libcpab import Cpab


def get_locnet():
    # Spatial transformer localization-network
    locnet = nn.Sequential(
        nn.Conv1d(1, 128, kernel_size=7),
        # nn.BatchNorm1d(128),
        nn.MaxPool1d(3, stride=2),
        nn.ReLU(True),
        nn.Conv1d(128, 64, kernel_size=9),
        # nn.BatchNorm1d(64),
        nn.MaxPool1d(3, stride=3),
        nn.ReLU(True),
        nn.Conv1d(64, 64, kernel_size=3),
        # nn.BatchNorm1d(),
        nn.MaxPool1d(3, stride=2),
        nn.ReLU(True),
        # GAP (when size=1) -
        # Note: While GAP allow the model size to remain fix w.r.t input length,
        # Temporal information is lost by the GAP operator.
        #nn.AdaptiveAvgPool1d(1),
    )
    return locnet

class DTAN(nn.Module):
    '''
    PyTroch nn.Module implementation of Diffeomorphic Temporal Alignment Nets [1]
    '''
    def __init__(self, input_shape, channels, tess=[6,], n_recurrence=1, zero_boundary=True, device='gpu'):
        '''

        Args:
            input_shape (int): signal length
            channels (int): number of channels
            tess (list): tessellation shape.
            n_recurrence (int): Number of recurrences for R-DTAN. Increasing the number of recurrences
                            Does not increase the number of parameters, but does the trainning time. Default is 1.
            zero_boundary (bool): Zero boundary (when True) for input X and transformed version X_T,
                                  sets X[0]=X_T[0] and X[n] = X_T[n]. Default is true.
            device: 'gpu' or 'cpu'
        '''
        super(DTAN, self).__init__()

        # init CPAB transformer
        self.T = Cpab(tess, backend='pytorch', device=device, zero_boundary=zero_boundary, volume_perservation=False)
        self.dim = self.T.get_theta_dim()
        self.n_recurrence = n_recurrence
        self.input_shape = input_shape # signal len
        self.channels = channels
        self.localization = get_locnet()
        self.fc_input_dim = self.get_conv_to_fc_dim()

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, self.dim),
            # Tanh constrains theta between -1 and 1
            nn.Tanh()
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[-2].bias.data.copy_(torch.clone(self.T.identity(epsilon=0.001).view(-1)))

    def get_conv_to_fc_dim(self):
        rand_tensor = torch.rand([1, self.channels, self.input_shape])
        out_tensor = self.localization(rand_tensor)
        conv_to_fc_dim = out_tensor.size(1)*out_tensor.size(2)
        #print("conv_to_fc_dim",conv_to_fc_dim, "full size", out_tensor.size())
        return conv_to_fc_dim

    # Spatial transformer network forward function
    def stn(self, x, return_theta=False):
        xs = self.localization(x)
        xs = xs.view(-1, self.fc_input_dim)
        theta = self.fc_loc(xs)
        x = self.T.transform_data(x, theta, outsize=(self.input_shape,))
        if not return_theta:
            return x
        else:
            return x, theta

    def forward(self, x, return_theta=False):
        # transform the input
        thetas = []
        for i in range(self.n_recurrence):
            if not return_theta:
                x = self.stn(x)
            else:
                x, theta = self.stn(x, return_theta)
                thetas.append(theta)
        if not return_theta:
            return x
        else:
            return x, thetas

    def get_basis(self):
        return self.T

# References:
# [1] - Diffeomorphic Temporal Alignment Nets (NeurIPS 2019)