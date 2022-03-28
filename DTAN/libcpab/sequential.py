#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 10:45:48 2019

@author: nsde
"""

#%%
from .cpab import Cpab

import matplotlib.pyplot as plt

#%%
class CpabSequential(object):
    ''' Helper class meant to make it easy to work with a sequence of transformers.
        Main method of the class are the transform_grid() and transform_data() that
        works similar to the methods of the core class.
        Example:
            T1 = cpab([2,2], ...)
            T2 = cpab([4,4], ...)
            theta1 = T1.sample_theta()
            theta2 = T2.sample_theta()
            T = SequentialCpab(T1, T2)
            data_trans = T.transform_data(some_data, theta1, theta2, outputshape)
    '''
    def __init__(self, *cpab):
        self.n_cpab = len(cpab)
        self.cpab = cpab
        
        # Assert that all cpab classes are valid
        for i in range(self.n_cpab):
            assert isinstance(self.cpab[i], Cpab), \
                ''' Class {0} is not a member of the cpab core class '''.format(i)
        
        # Assert that all cpab classes have same dimensionality
        self.ndim = self.cpab[0].params.ndim
        for i in range(1, self.n_cpab):
            assert self.ndim == self.cpab[i].params.ndim, \
                ''' Mismatching dimensionality of transformers. Transformer 1
                have dimensionality {0} but transformer {1} have dimensionality
                {2}'''.format(self.ndim, i+1, self.cpab[i].params.ndim)
                
        # Assert that all cpab classes have same backend
        self.backend = self.cpab[0].backend
        self.backend_name = self.cpab[0].backend_name
        for i in range(1, self.n_cpab):
            assert self.backend_name == self.cpab[i].backend_name, \
                ''' Mismatch in backend. Transformer 1 have backend {0} but
                transformer {1} have backend {2}'''.format(
                self.backend_name, i+1, self.cpab[i].backend_name)
        
                
    #%%
    def get_theta_dim(self):
        return [c.get_theta_dim for c in self.cpab]
    
    #%%
    def get_params(self):
        return [c.get_params() for c in self.cpab]
    
    #%%
    def get_basis(self):
        return [c.get_basis() for c in self.cpab]
    
    #%%
    def uniform_meshgrid(self, n_points):
        return self.cpab[0].uniform_meshgrid(n_points)
    
    #%%
    def sample_transformation(self, n_sample, means=None, covs=None):
        if means==None:
            means = self.n_cpab * [None]
        else:
            assert len(means)==self.n_cpab, ''' The number of supplied means
                should be equal to the number of transformations '''
        if covs==None:
            covs = self.n_cpab * [None]
        else:
            assert len(covs)==self.n_cpab, ''' The number of supplied covariances
                should be equal to the number of transformations '''

        return [c.sample_transformation(n_sample, mean, cov) for c,mean,cov in 
                zip(self.cpab, means, covs)]
        
    #%%
    def identity(self, n_sample, epsilon=0):
        return [c.identity(n_sample, epsilon) for c in self.cpab]
    
    #%%
    def transform_grid(self, grid, thetas, output_all=False):
        # Check shapes of thetas
        self._assert_theta_shape(thetas)

        if not output_all:
            # Transform in sequence
            for i in range(self.n_cpab):
                grid = self.cpab[i].transform_grid(grid, thetas[i])
        else:
            grid = [self.cpab[0].transform_grid(grid, thetas[0])]
            for i in range(1, self.n_cpab):
                grid.append(self.cpab[i].transform_grid(grid[-1], thetas[i]))
        
        return grid
        
    #%%
    def transform_data(self, data, thetas, outsize, output_all=False):
        # Check shapes of thetas
        self._assert_theta_shape(thetas)
        
        if not output_all:
            # Transform in sequence
            grid = self.uniform_meshgrid(outsize)
            grid_t = self.transform_grid(grid, thetas, output_all=output_all)
            
            # Interpolate using final grid
            data_t = self.cpab[-1].interpolate(data, grid_t, outsize)
            return data_t
        else:
            # Transform in sequence
            grid = self.uniform_meshgrid(outsize)
            grid_t = self.transform_grid(grid, thetas, output_all=output_all)
            
            # Interpolate all grids
            data_t = [self.cpab[0].interpolate(data, grid_t[0], outsize)]
            for i in range(1, self.n_cpab):
                data_t.append(self.cpab[i].interpolate(data, grid_t[i], outsize))
            return data_t
    
    #%%
    def _assert_theta_shape(self, thetas):
        n_theta = len(thetas)
        assert n_theta == self.n_cpab, \
            ''' Number of parametrizations needed are {0}'''.format(self.n_trans)
        batch_size = thetas[0].shape[0]
        for i in range(1, n_theta):
            assert batch_size == thetas[i].shape[0], ''' Batch size should be the
                same for all theta's '''
    
    #%%
    def __repr__(self):
        for i in range(self.n_cpab):
            print("======= Transformer {0} ======= ".format(i+1))
            print(self.cpab[i])