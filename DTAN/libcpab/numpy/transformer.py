# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 10:27:16 2018

@author: nsde
"""

#%%
import numpy as np
from scipy.linalg import expm
from .findcellidx import findcellidx

#%%
compiled = False

#%%
def CPAB_transformer(points, theta, params):
    if compiled: return CPAB_transformer_fast(points, theta, params)
    else: return CPAB_transformer_slow(points, theta, params)

#%%
def CPAB_transformer_slow(points, theta, params):
    # Problem parameters
    n_theta = theta.shape[0]
    n_points = points.shape[-1]
    
    # Create homogenous coordinates
    ones = np.ones((n_theta, 1, n_points))
    if len(points.shape)==2: # tile if 2D grid
        newpoints = np.tile(points, [n_theta, 1, 1]) # [n_theta, ndim, n_points]
    else:
        newpoints = points
    newpoints = np.concatenate((newpoints, ones), axis=1) # [n_theta, ndim+1, n_points]
    newpoints = np.transpose(newpoints, (0, 2, 1)) # [n_theta, n_points, ndim+1]
    newpoints = np.reshape(newpoints, (-1, params.ndim+1)) #[n_theta*n_points, ndim+1]]
    newpoints = newpoints[:,:,None] # [n_theta*n_points, ndim+1, 1]
    
    # Get velocity fields
    B = params.basis
    Avees = np.matmul(B, theta.T)
    As = Avees.T.reshape(n_theta*params.nC, *params.Ashape)
    zero_row = np.zeros((n_theta*params.nC, 1, params.ndim+1))
    AsSquare = np.concatenate([As, zero_row], axis=1)
    
    # Take matrix exponential
    dT = 1.0 / params.nstepsolver
    Trels = np.stack([expm(dT*array) for array in AsSquare])
    
    # Take care of batch effect
    batch_idx = params.nC*(np.ones((n_points, n_theta)) * np.arange(n_theta))
    batch_idx = batch_idx.T.flatten().astype(np.int32)
    
    # Do integration
    for i in range(params.nstepsolver):
        idx = findcellidx(params.ndim, newpoints[:,:,0].T, params.nc) + batch_idx
        Tidx = Trels[idx]
        newpoints = np.matmul(Tidx, newpoints)
    
    newpoints = newpoints.squeeze()[:,:params.ndim].T
    newpoints = np.transpose(newpoints.reshape(params.ndim, n_theta, n_points), (1,0,2))
    return newpoints

#%%
def CPAB_transformer_fast(points, theta):
    # TODO: jit compile cpp code into callable python code
    pass