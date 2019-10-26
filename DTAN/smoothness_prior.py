"""
Created on Oct  2019

author: ronsha
"""


# From other libraries
import numpy as np


# From tensorflow
import tensorflow as tf
from tensorflow.python.keras import backend as K

# From local files
from helper.tf_functions import tf_dist_mat, tf_plot_graphs


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


# Domain space is [0,1]^dim where 0.5 is the origin

def smoothness_norm(T, theta, lambda_smooth=0.5, lambda_var=0.1, print_info=False):
    D, d = T.get_basis().shape
    B = T.get_basis()
    nC = d + 1  # = Tess size
    n = 1  # num sampples?

    # Convert type
    B = tf.cast(B, tf.float32)
    theta = tf.cast(theta, tf.float32)
    theta_T = tf.transpose(theta)

    # for plotting, {"title": item_to_plot/show}
    covariance_to_plot = {}
    items_to_plot = {}

    # Distance between centers
    centers = tf.lin_space(-1., 1., D)  # from 0 to 1 with nC steps
    centers = tf.expand_dims(centers, 1)

    # calculate the distance
    dists = tf_dist_mat(centers) # DxD

    # # scale the distance
    # for x>0, e^(-x^2) decays very fast from 1 to 0

    cov_avees = tf.exp(-(dists / lambda_smooth))
    cov_avees *= (cov_avees * (lambda_var * D) ** 2)

    # Calculate covariance matrix for theta space
    B_T = tf.transpose(B)
    cov_cpa = tf.matmul(B_T, tf.matmul(cov_avees, B))
    precision_theta = tf.linalg.inv(cov_cpa)

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
    theta = tf.squeeze(theta)
    theta_T = tf.transpose(theta)
    smooth_norm = tf.matmul(theta, tf.matmul(precision_theta, theta_T))
    smooth_norm = 0.1*tf.reduce_mean(smooth_norm)

    return smooth_norm