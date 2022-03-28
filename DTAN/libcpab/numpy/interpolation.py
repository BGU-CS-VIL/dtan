# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 10:26:30 2018

@author: nsde
"""

#%%
import numpy as np

#%%
def interpolate(ndim, data, grid, outsize):
    if ndim==1: return interpolate1D(data, grid, outsize)
    elif ndim==2: return interpolate2D(data, grid, outsize)
    elif ndim==3: return interpolate3D(data, grid, outsize)

#%%    
def interpolate1D(data, grid, outsize):
    # Problem size
    n_batch = data.shape[0]
    width = data.shape[1]
    n_channels = data.shape[2]
    out_width = outsize[0]
    
    # Extract points
    x = grid[:,0].flatten()

    # Scale to domain
    x = x * (width-1)
    
    # Do sampling
    x0 = np.floor(x).astype(np.int32); x1 = x0+1
    
    # Clip values
    x0 = np.clip(x0, 0, width-1)
    x1 = np.clip(x1, 0, width-1)
    
    # Batch effect
    batch_size = out_width
    batch_idx = np.arange(n_batch).repeat(batch_size)
    
    # Index
    c0 = data[batch_idx, x0, :]
    c1 = data[batch_idx, x1, :]
    
    # Interpolation weights
    xd = (x-x0.astype(np.float32)).reshape((-1,1))
    
    # Do interpolation
    c = c0*(1-xd) + c1*xd
    
    # Reshape
    new_data = np.reshape(c, (n_batch, out_width, n_channels))
    return new_data
    
    
#%%    
def interpolate2D(data, grid, outsize):
    # Problem size
    n_batch = data.shape[0]
    width = data.shape[1]
    height = data.shape[2]
    n_channels = data.shape[3]
    out_width, out_height = outsize
    
    # Extract points
    x = grid[:,0].flatten()
    y = grid[:,1].flatten()
    
    # Scale to domain
    x = x * (width-1)
    y = y * (height-1)
    
    # Do sampling
    x0 = np.floor(x).astype(np.int32); x1 = x0+1
    y0 = np.floor(y).astype(np.int32); y1 = y0+1
    
    # Clip values
    x0 = np.clip(x0, 0, width-1)
    x1 = np.clip(x1, 0, width-1)
    y0 = np.clip(y0, 0, height-1)
    y1 = np.clip(y1, 0, height-1)
    
    # Batch effect
    batch_size = out_width*out_height
    batch_idx = np.arange(n_batch).repeat(batch_size)
    
    # Index
    c00 = data[batch_idx, x0, y0, :]
    c01 = data[batch_idx, x0, y1, :]
    c10 = data[batch_idx, x1, y0, :]
    c11 = data[batch_idx, x1, y1, :]
    
    # Interpolation weights
    xd = (x-x0.astype(np.float32)).reshape((-1,1))
    yd = (y-y0.astype(np.float32)).reshape((-1,1))
    
    # Do interpolation
    c0 = c00*(1-xd) + c10*xd
    c1 = c01*(1-xd) + c11*xd
    c = c0*(1-yd) + c1*yd
    
    # Reshape
    new_data = np.reshape(c, (n_batch, out_height, out_width, n_channels))
    new_data = np.transpose(new_data, (0, 2, 1, 3))
    return new_data
    
#%%    
def interpolate3D(data, grid, outsize):
    # Problem size
    n_batch = data.shape[0]
    width = data.shape[1]
    height = data.shape[2]
    depth = data.shape[3]
    n_channels = data.shape[4]    
    out_width, out_height, out_depth = outsize
    
    # Extract points
    x = grid[:,0].flatten()
    y = grid[:,1].flatten()
    z = grid[:,2].flatten()
    
    # Scale to domain
    x = x * (width-1)
    y = y * (height-1)
    z = z * (depth-1)
    
    # Do sampling
    x0 = np.floor(x).astype(np.int32); x1 = x0+1
    y0 = np.floor(y).astype(np.int32); y1 = y0+1
    z0 = np.floor(z).astype(np.int32); z1 = z0+1
    
    # Clip values
    x0 = np.clip(x0, 0, width-1)
    x1 = np.clip(x1, 0, width-1)
    y0 = np.clip(y0, 0, height-1)
    y1 = np.clip(y1, 0, height-1)
    z0 = np.clip(z0, 0, depth-1)
    z1 = np.clip(z1, 0, depth-1)
    
    # Batch effect
    batch_size = out_width*out_height*out_depth
    batch_idx = np.arange(n_batch).repeat(batch_size)
    
    # Index
    c000 = data[batch_idx, x0, y0, z0, :]
    c001 = data[batch_idx, x0, y0, z1, :]
    c010 = data[batch_idx, x0, y1, z0, :]
    c011 = data[batch_idx, x0, y1, z1, :]
    c100 = data[batch_idx, x1, y0, z0, :]
    c101 = data[batch_idx, x1, y0, z1, :]
    c110 = data[batch_idx, x1, y1, z0, :]
    c111 = data[batch_idx, x1, y1, z1, :]
    
    # Interpolation weights
    xd = (x-x0.astype(np.float32)).reshape((-1,1))
    yd = (y-y0.astype(np.float32)).reshape((-1,1))
    zd = (z-z0.astype(np.float32)).reshape((-1,1))
    
    # Do interpolation
    c00 = c000*(1-xd) + c100*xd
    c01 = c001*(1-xd) + c101*xd
    c10 = c010*(1-xd) + c110*xd
    c11 = c011*(1-xd) + c111*xd
    c0 = c00*(1-yd) + c10*yd
    c1 = c01*(1-yd) + c11*yd
    c = c0*(1-zd) + c1*zd
    
    # Reshape
    new_data = np.reshape(c, (n_batch, out_depth, out_height, out_width, n_channels))
    new_data = np.transpose(new_data, (0, 3, 2, 1, 4))
    return new_data
        
