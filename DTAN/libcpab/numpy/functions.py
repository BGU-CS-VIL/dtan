# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 16:01:52 2018

@author: nsde
"""

#%%
import numpy as np
from .interpolation import interpolate
from .transformer import CPAB_transformer as transformer
from .findcellidx import findcellidx

#%%
def assert_version():
    numbers = np.__version__.split('.')
    version = float(numbers[0] + '.' + numbers[1])
    assert version >= 1.14, \
        ''' You are using a older installation of numpy, please install 1.15.
            or newer '''

#%%
def to(x, dtype=np.float32, device=None): 
    return np.array(x)

#%%
def tonumpy(x):
    return x

#%%
def check_device(x, device_name):
    return True # always return true, because device can only be cpu

#%%
def backend_type():
    return np.ndarray

#%%
def pdist(mat):
    norm = np.sum(mat * mat, 1)
    norm = np.reshape(norm, (-1, 1))
    D = norm - 2*np.matmul(mat, mat.T) + norm.T
    return D

#%%
def norm(x):
    return np.linalg.norm(x)

#%%
def matmul(x,y):
    return np.matmul(x,y)

#%%
def transpose(x):
    return x.T

#%%
def exp(x):
    return np.exp(x)

#%%
def zeros(*s):
    return np.zeros(*s)
    
#%%
def ones(*s):
    return np.ones(*s)

#%%
def arange(x):
    return np.arange(x)
    
#%%
def repeat(x, reps):
    return np.repeat(x, reps)

#%%
def batch_repeat(x, n_batch):
    return np.repeat(x[None], n_batch, axis=0)

#%%
def maximum(x):
    return np.max(x)
    
#%%
def sample_transformation(d, n_sample=1, mean=None, cov=None, device='cpu'):
    mean = np.zeros(d, dtype=np.float32) if mean is None else mean
    cov = np.eye(d, dtype=np.float32) if cov is None else cov
    samples = np.random.multivariate_normal(mean, cov, size=n_sample)
    return samples

#%%
def identity(d, n_sample=1, epsilon=0, device='cpu'):
    assert epsilon>=0, "epsilon need to be larger than or 0"
    return np.zeros((n_sample, d), dtype=np.float32) + epsilon

#%%
def uniform_meshgrid(ndim, domain_min, domain_max, n_points, device='cpu'):
    lin = [np.linspace(domain_min[i], domain_max[i], n_points[i]) for i in range(ndim)]
    mesh = np.meshgrid(*lin[::-1], indexing='ij')
    grid = np.vstack([array.flatten() for array in mesh[::-1]])
    return grid

#%%
def calc_vectorfield(grid, theta, params):
    # Calculate velocity fields
    Avees = np.matmul(params.basis, theta.flatten())
    As = np.reshape(Avees, (params.nC, *params.Ashape))
    
    # Find cell index
    idx = findcellidx(params.ndim, grid, params.nc)
    
    # Do indexing
    Aidx = As[idx]
    
    # Convert to homogeneous coordinates
    grid = np.concatenate((grid, np.ones((1, grid.shape[1]))), axis=0)
    grid = np.transpose(grid[None], axes=[2,1,0])
    
    # Do matrix multiplication
    v = np.matmul(Aidx, grid)
    return np.transpose(v[:,:,0]) # output: [ndim, nP] 
    