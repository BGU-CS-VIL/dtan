"""
Created on Oct  2019
author: ronsha
"""

# From other libraries
import numpy as np

import torch


"""
Smoothness Norm:
Computes the smoothness and L2 norm for a certian paramterzation of theta for cpab transformer T. 
1. Smoothness prior: penalize low correlation of theta between close cells in the tessellation. 
   It is computed by building a (D,D) covariance correlations decay with inter-cell distances.

2. L2 norm: penalize large values of theta
Arguments:
    theta: current parametrization of the transformation.
    T: cpab class of transformer type
    scale_spatial: smoothness regularization strength
    scale_value: L2 regularization strength
    print_info: for debugging, will probably be removed. 
Returns:
    Smoothness norm: high values indicates lack of smoothness. To be added to the loss as a regularization. 
"""


def torch_dist_mat(centers):
    '''
    Produces an NxN  dist matrix D,  from vector (centers) of size N
    Diagnoal = 0, each entry j, represent the distance from the diagonal
    dictated by the centers vector input
    '''
    times = centers.shape  # Torch.Tensor([n], shape=(1,), dtype=int32)

    # centers_grid tile of shape N,N, each row = centers
    centers_grid = centers.repeat(times[0],1)
    dist_matrix = torch.abs(centers_grid - torch.transpose(centers_grid, 0, 1))
    return dist_matrix


# Domain space is [0,1]^dim where 0.5 is the origin
def smoothness_norm(T, theta, lambda_smooth=0.5, lambda_var=0.1, print_info=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D, d = T.get_basis().shape
    B = T.get_basis()
    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B).float()
        B = B.to(device)

    nC = d + 1  # = Tess size
    n = 1  # num sampples?

    # Convert type
    #B = tf.cast(B, tf.float32)
    #theta = tf.cast(theta, tf.float32)
    theta_T = torch.transpose(theta, 0, 1)

    # for plotting, {"title": item_to_plot/show}
    covariance_to_plot = {}
    items_to_plot = {}

    # Distance between centers
    centers = torch.linspace(-1., 1., D).to(device)  # from 0 to 1 with nC steps

    # calculate the distance
    dists = torch_dist_mat(centers)  # DxD

    # # scale the distance
    # for x>0, e^(-x^2) decays very fast from 1 to 0

    cov_avees = torch.exp(-(dists / lambda_smooth))
    cov_avees *= (cov_avees * (lambda_var * D) ** 2)

    B_T = torch.transpose(B, 0, 1)
    cov_cpa = torch.matmul(B_T, torch.matmul(cov_avees, B))
    precision_theta = torch.inverse(cov_cpa)

    if print_info:
        print("Info: ")
        print("    Distance between centers shape:", dists.shape)
        print("    B shape:", B.shape)
        print("    D shape:", D)
        print("    d shape:", d)
        print("    cov_avess shape:", cov_avees.shape)
        print("    cov_cpa shape:", cov_cpa.shape)
        print("    theta shape:", theta.shape)

    # plot velocity field
    plot_velocity_field = False
    if plot_velocity_field:
        nb_points = 1000
        points = T.uniform_meshgrid([nb_points for i in range(T._ndim)])
        vector_field = T.calc_vectorfield(points, theta_T.eval())  # tf.transpose(theta)
        items_to_plot["CPA Velocity Field for theta with prior"] = vector_field
    items_to_plot["Theta values for: theta with prior"] = theta_T

    # Calculate smoothness norm
    theta_T = torch.transpose(theta, 0, 1)
    smooth_norm = torch.matmul(theta, torch.matmul(precision_theta, theta_T))
    smooth_norm = torch.mean(smooth_norm)

    return smooth_norm

