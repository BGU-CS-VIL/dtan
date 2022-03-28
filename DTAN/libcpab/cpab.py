#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:34:36 2018

@author: nsde
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from .core.utility import params, get_dir, create_dir
from .core.tesselation import Tesselation1D, Tesselation2D, Tesselation3D

#%%
class Cpab(object):
    """ Core class for this library. This class contains all the information
        about the tesselation, transformation ect. The user is not meant to
        use anything else than this specific class.
        
    Arguments:
        tess_size: list, with the number of cells in each dimension
        
        backend: string, computational backend to use. Choose between 
            "numpy" (default), "pytorch" or "tensorflow"
        
        device: string, either "cpu" (default) or "gpu". For the numpy backend
            only the "cpu" option is valid
        
        zero_boundary: bool, determines is the velocity at the boundary is zero 
        
        volume_perservation: bool, determine if the transformation is 
            volume perservating
            
        override: bool, if true a new basis will always be created and saved,
            when the class is called
        
    Methods:
        @get_theta_dim
        @get_params
        @get_bases
        @uniform_meshgrid
        @sample_transformation
        @sample_transformation_with_smooth_prior
        @identity
        @transform_grid
        @interpolate
        @transform_data
        @calc_vectorfield
        @visualize_vectorfield
        @visualize_tesselation
        @visualize_deformgrid
    """
    def __init__(self, 
                 tess_size,
                 backend = 'numpy',
                 device = 'cpu', 
                 zero_boundary=True,
                 volume_perservation=False,
                 override=False):
        # Check input
        self._check_input(tess_size, backend, device, 
                          zero_boundary, volume_perservation, override)
        
        # Parameters
        self.params = params()
        self.params.nc = tess_size
        self.params.ndim = len(tess_size)
        self.params.Ashape = [self.params.ndim, self.params.ndim+1]
        self.params.valid_outside = not(zero_boundary)
        self.params.zero_boundary = zero_boundary
        self.params.volume_perservation = volume_perservation
        self.params.domain_max = [1 for e in self.params.nc]
        self.params.domain_min = [0 for e in self.params.nc]
        self.params.inc = [(self.params.domain_max[i] - self.params.domain_min[i]) / 
                           self.params.nc[i] for i in range(self.params.ndim)]
        self.params.nstepsolver = 50
        self.params.numeric_grad = False
        self.params.use_slow = False
        
        # For saving the basis
        self._dir = get_dir(__file__) + '/../basis_files/'
        create_dir(self._dir)
        
        # Specific for the different dims
        if self.params.ndim == 1:
            self.params.nC = self.params.nc[0]
            self.params.params_pr_cell = 2
            tesselation = Tesselation1D
        elif self.params.ndim == 2:
            self.params.nC = 4*np.prod(self.params.nc)
            self.params.params_pr_cell = 6
            tesselation = Tesselation2D
        elif self.params.ndim == 3:
            self.params.nC = 5*np.prod(self.params.nc)
            self.params.params_pr_cell = 12
            tesselation = Tesselation3D
            
        # Initialize tesselation
        self.tesselation = tesselation(self.params.nc, self.params.domain_min, 
                                       self.params.domain_max, self.params.zero_boundary, 
                                       self.params.volume_perservation,
                                       self._dir, override)
        
        # Extract parameters from tesselation
        self.params.constrain_mat = self.tesselation.L
        self.params.basis = self.tesselation.B
        self.params.D, self.params.d = self.params.basis.shape
                
        # Load backend and set device
        self.backend_name = backend
        if self.backend_name == 'numpy':
            from .numpy import functions as backend
        elif self.backend_name == 'pytorch':
            from .pytorch import functions as backend
        self.backend = backend
        self.device = device.lower()
        
        # Assert that we have a recent version of the backend
        self.backend.assert_version()
        
    #%%
    def get_theta_dim(self):
        """ Method that returns the dimensionality of the transformation"""
        return self.params.d
    
    #%%
    def get_params(self):
        """ Returns a class with all parameters for the transformation """
        return self.params
    
    #%%
    def get_basis(self):
        """ Method that return the basis of transformation"""
        return self.params.basis
    
    #%%
    def set_solver_params(self, nstepsolver=50, numeric_grad=False, use_slow=False):
        """ Function for setting parameters that controls parameters of the
            integration algorithm. Only use if you know what you do.
        Arguments:
            nstepsolver: int, number of iterations to take in integration. Higher
                number give better approximations but takes longer time to compute
            numeric_grad: bool, determines if we should use the analytical grad
                or numeric grad for gradient computations
            use_slow: bool, determine if the integration should be done using the
                pure "python" version of each backend
        """
        assert nstepsolver > 0, '''nstepsolver must be a positive number'''
        assert type(nstepsolver) == int, '''nstepsolver must be integer'''
        assert type(numeric_grad) == bool, '''numeric_grad must be bool'''
        assert type(use_slow) == bool, '''use_slow must be bool'''
        self.params.nstepsolver = nstepsolver
        self.params.numeric_grad = numeric_grad
        self.params.use_slow = use_slow
        
    #%%    
    def uniform_meshgrid(self, n_points):
        """ Constructs a meshgrid 
        Arguments:
            n_points: list, number of points in each dimension
        Output:
            grid: [ndim, nP] matrix of points, where nP = product(n_points)
        """
        return self.backend.uniform_meshgrid(self.params.ndim,
                                             self.params.domain_min,
                                             self.params.domain_max,
                                             n_points, self.device)
      
    #%%
    def sample_transformation(self, n_sample=1, mean=None, cov=None):
        """ Method for sampling transformation from simply multivariate gaussian
            As default the method will sample from a standard normal
        Arguments:
            n_sample: integer, number of transformations to sample
            mean: [d,] vector, mean of multivariate gaussian
            cov: [d,d] matrix, covariance of multivariate gaussian
        Output:
            samples: [n_sample, d] matrix. Each row is a independent sample from
                a multivariate gaussian
        """
        if mean is not None: self._check_type(mean); self._check_device(mean)
        if cov is not None: self._check_type(cov); self._check_device(cov)
        samples = self.backend.sample_transformation(self.params.d, n_sample, 
                                                     mean, cov, self.device)
        return self.backend.to(samples, device=self.device)
        
    
    #%%
    def sample_transformation_with_prior(self, n_sample=1, mean=None, 
                                         length_scale=0.1, output_variance=1):
        """ Function for sampling smooth transformations. The smoothness is determined
            by the distance between cell centers. The closer the cells are to each other,
            the more the cell parameters should correlate -> smooth transistion in
            parameters. The covariance in the D-space is calculated using the
            squared exponential kernel.
                
        Arguments:
            n_sample: integer, number of transformation to sample
            mean: [d,] vector, mean of multivariate gaussian
            length_scale: float>0, determines how fast the covariance declines 
                between the cells 
            output_variance: float>0, determines the overall variance from the mean
        Output:
            samples: [n_sample, d] matrix. Each row is a independen sample from
                a multivariate gaussian
        """
        
        # Get cell centers and convert to backend type
        centers = self.backend.to(self.tesselation.get_cell_centers(), device=self.device)
        
        # Get distance between cell centers
        dist = self.backend.pdist(centers)
        
        # Make into a covariance matrix between parameters
        ppc = self.params.params_pr_cell
        cov_init = self.backend.zeros(self.params.D, self.params.D, device=self.device)
        
        for i in range(self.params.nC):
            for j in range(self.params.nC):
                # Make block matrix with large values
                block = 100*self.backend.maximum(dist)*self.backend.ones(ppc, ppc)
                # Fill in diagonal with actual values
                block[self.backend.arange(ppc), self.backend.arange(ppc)] = \
                    self.backend.repeat(dist[i,j], ppc)
                # Fill block into the large covariance
                cov_init[ppc*i:ppc*(i+1), ppc*j:ppc*(j+1)] = block
        
        # Squared exponential kernel
        cov_avees = output_variance**2 * self.backend.exp(-(cov_init / (2*length_scale**2)))

        # Transform covariance to theta space
        B = self.backend.to(self.params.basis, self.device)
        B_t = self.backend.transpose(B)
        cov_theta = self.backend.matmul(B_t, self.backend.matmul(cov_avees, B))
        
        # Sample
        samples = self.sample_transformation(n_sample, mean=mean, cov=cov_theta)
        return samples
    
    #%%
    def identity(self, n_sample=1, epsilon=0):
        """ Method for getting the identity parameters for the identity 
            transformation (vector of zeros) 
        Arguments:
            n_sample: integer, number of transformations to sample
            epsilon: float>0, small number to add to the identity transformation
                for stability during training
        Output:
            samples: [n_sample, d] matrix. Each row is a sample    
        """
        return self.backend.identity(self.params.d, n_sample, epsilon, self.device)
    
    #%%
    def transform_grid(self, grid, theta):
        """ Main method of the class. Integrates the grid using the parametrization
            in theta.
        Arguments:
            grid: [ndim, n_points] matrix or [n_batch, ndim, n_points] tensor i.e.
                either a single grid for all theta values, or a grid for each theta
                value
            theta: [n_batch, d] matrix,
        Output:
            transformed_grid: [n_batch, ndim, n_points] tensor, with the transformed
                grid. The slice transformed_grid[i] corresponds to the grid being
                transformed by theta[i]
        """
        self._check_type(grid); self._check_device(grid)
        self._check_type(theta); self._check_device(theta)
        if len(grid.shape) == 3: # check that grid and theta can broadcastes together
            assert grid.shape[0] == theta.shape[0], '''When passing a 3D grid, expects
                the first dimension to be of same length as the first dimension of
                theta'''
        transformed_grid = self.backend.transformer(grid, theta, self.params)
        return transformed_grid
    
    #%%    
    def interpolate(self, data, grid, outsize):
        """ Linear interpolation method
        Arguments:
            data: [n_batch, *data_shape] tensor, with input data. The format of
                the data_shape depends on the dimension of the data AND the
                backend that is being used. In tensorflow and numpy:
                    In 1D: [n_batch, number_of_features, n_channels]
                    In 2D: [n_batch, width, height, n_channels]
                    In 3D: [n_batch, width, height, depth, n_channels]
                In pytorch:
                    In 1D: [n_batch, n_channels, number_of_features]
                    In 2D: [n_batch, n_channels, width, height]
                    In 3D: [n_batch, n_channels, width, height, depth]
            grid: [n_batch, ndim, n_points] tensor with grid points that are 
                used to interpolate the data
            outsize: list, with number of points in the output
        Output:
            interlated: [n_batch, *outsize] tensor with the interpolated data
        """            
        self._check_type(data); self._check_device(data)
        self._check_type(grid); self._check_device(grid)
        return self.backend.interpolate(self.params.ndim, data, grid, outsize)
    
    #%%
    def transform_data(self, data, theta, outsize):
        """ Combination of the transform_grid and interpolate methods for easy
            transformation of data.
        Arguments:
            data: [n_batch, *data_shape] tensor, with input data. The format of
                the data_shape depends on the dimension of the data AND the
                backend that is being used. In tensorflow and numpy:
                    In 1D: [n_batch, number_of_features, n_channels]
                    In 2D: [n_batch, width, height, n_channels]
                    In 3D: [n_batch, width, height, depth, n_channels]
                In pytorch:
                    In 1D: [n_batch, n_channels, number_of_features]
                    In 2D: [n_batch, n_channels, width, height]
                    In 3D: [n_batch, n_channels, width, height, depth]
            theta: [n_batch, d] matrix with transformation parameters. Each row
                correspond to a transformation.
            outsize: list, number of points in each direction that is transformed
                and interpolated
        Output:
            data_t: [n_batch, *outsize] tensor, transformed and interpolated data
        """

        self._check_type(data); self._check_device(data)
        self._check_type(theta); self._check_device(theta)
        grid = self.uniform_meshgrid(outsize)
        grid_t = self.transform_grid(grid, theta)
        data_t = self.interpolate(data, grid_t, outsize)
        return data_t
    
    #%%
    def calc_vectorfield(self, grid, theta):
        """ For each point in grid, calculate the velocity of the point based
            on the parametrization in theta
        Arguments:
            grid: [ndim, nP] matrix, with points
            theta: [1, d] single parametrization vector
        Output:    
            v: [ndim, nP] matrix, with velocity vectors for each point
        """
        self._check_type(grid); self._check_device(grid)
        self._check_type(theta); self._check_device(theta)
        v = self.backend.calc_vectorfield(grid, theta, self.params)
        return v
    
    #%%
    def visualize_vectorfield(self, theta, nb_points = 50, fig = plt.figure()):
        """ Utility function that helps visualize the vectorfield for a specific
            parametrization vector theta 
        Arguments:    
            theta: [1, d] single parametrization vector
            nb_points: number of points in each dimension to plot i.e. in 2D
                with nb_points=50 the function will plot 50*50=2500 arrows!
            fig: matplotlib figure handle
        Output:
            plot: handle to quiver plot
        """
        self._check_type(theta)
        
        # Calculate vectorfield and convert to numpy
        grid = self.uniform_meshgrid([nb_points for _ in range(self.params.ndim)])
        v = self.calc_vectorfield(grid, theta)
        v = self.backend.tonumpy(v)
        grid = self.backend.tonumpy(grid)
        
        # Plot
        if self.params.ndim == 1:
            ax = fig.add_subplot(111)
            plot = ax.quiver(grid[0,:], np.zeros_like(grid), v, np.zeros_like(v), units='xy')
            ax.set_xlim(self.params.domain_min[0], self.params.domain_max[0])
        elif self.params.ndim == 2:
            ax = fig.add_subplot(111)
            plot = ax.quiver(grid[0,:], grid[1,:], v[0,:], v[1,:], units='xy')
            ax.set_xlim(self.params.domain_min[0], self.params.domain_max[0])
            ax.set_ylim(self.params.domain_min[1], self.params.domain_max[1])            
        elif self.params.ndim==3:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
            plot = ax.quiver(grid[0,:], grid[1,:], grid[2,:], v[0,:], v[1,:], v[2,:],
                             length=0.3, arrow_length_ratio=0.5)
            ax.set_xlim3d(self.params.domain_min[0], self.params.domain_max[0])
            ax.set_ylim3d(self.params.domain_min[1], self.params.domain_max[1])
            ax.set_zlim3d(self.params.domain_min[2], self.params.domain_max[2])
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        plt.axis('equal')
        plt.title('Velocity field')
        return plot
    
    #%%
    def visualize_deformgrid(self, theta, nb_lines = 10, nb_points= 1000, fig = plt.figure()):
        """ Utility function that helps visualize a deformation. Currently
            only implemented in 2D.
        Arguments:
            theta: [1, d] single parametrization vector
            nb_lines: int, number of lines in x/y direction
            nb_points: int, number of points on each line
            fig: matplotlib figure handle
        Output:
            plot: list of plot handles to lines
        """
        if self.params.ndim == 2:
            x = np.linspace(0,1,nb_lines)
            y = np.linspace(0,1,nb_lines)
            plots = []
            for i in range(nb_lines):
                xx = x[i]*np.ones((1,nb_points))
                yy = np.linspace(0,1,nb_points).reshape(1,nb_points)
                grid = np.concatenate((xx, yy), axis=0)
                grid = self.transform_grid(grid, theta)[0]
                plot = plt.plot(grid[0], grid[1], '-k')
                plots.append(plot)
      
            for i in range(nb_lines):
                xx = np.linspace(0,1,nb_points).reshape(1,nb_points)
                yy = y[i]*np.ones((1,nb_points))
                grid = np.concatenate((xx, yy), axis=0)
                grid = self.transform_grid(grid, theta)[0]
                plot = plt.plot(grid[0], grid[1], '-k')            
                plots.append(plot)
            return plots
        else:
            raise NotImplementedError('This is only implemented for 2D domain')
    
    #%%
    def visualize_tesselation(self, nb_points = 50, show_outside=False, fig=plt.figure()):
        """ Utility function that helps visualize the tesselation.
        Arguments:
            nb_points: number of points in each dimension
            show_outside: if true, will sample points outside the normal [0,1]^ndim
                domain to show how the tesselation (or in fact the findcellidx)
                function extends to outside domain.
            fig: matplotlib figure handle
        Output:
            plot: handle to tesselation plot
        """
        if show_outside:
            domain_size = [self.params.domain_max[i] - self.params.domain_min[i] 
                           for i in range(self.params.ndim)]
            domain_min = [self.params.domain_min[i]-domain_size[i]/10 
                          for i in range(self.params.ndim)]
            domain_max = [self.params.domain_max[i]+domain_size[i]/10 
                          for i in range(self.params.ndim)]
            grid = self.backend.uniform_meshgrid(self.params.ndim, domain_min, 
                        domain_max, [nb_points for _ in range(self.params.ndim)])
        else:
            grid = self.uniform_meshgrid([nb_points for _ in range(self.params.ndim)])
        
        # Find cellindex and convert to numpy
        idx = self.backend.findcellidx(self.params.ndim, grid, self.params.nc)
        idx = self.backend.tonumpy(idx)
        grid = self.backend.tonumpy(grid)
        
        # Plot
        if self.params.ndim == 1:
            ax = fig.add_subplot(111)
            plot = ax.scatter(grid.flatten(), np.zeros_like(grid).flatten(), c=idx)
        elif self.params.ndim == 2:
            ax = fig.add_subplot(111)
            plot = ax.imshow(idx.reshape(self.params.ndim*[nb_points]))
        elif self.params.ndim == 3:
            import matplotlib.animation as animation
            idx = idx.reshape(self.params.ndim*[nb_points])
            im = plt.imshow(idx[0,:,:], animated=True)
            def update(frames):
                im.set_array(idx[frames,:,:])
                return im,
            plot = animation.FuncAnimation(fig, update, frames=nb_points, blit=True)
            cbar = plt.colorbar()
            cbar.set_clim(idx.min(), idx.max())
            cbar.update_ticks()
            
        plt.axis('equal')
        plt.title('Tesselation ' + str(self.params.nc))
        return plot
    
    #%%
    def _check_input(self, tess_size, backend, device, 
                     zero_boundary, volume_perservation, override):
        """ Utility function used to check the input to the class.
            Not meant to be called by the user. """
        assert len(tess_size) > 0 and len(tess_size) <= 3, \
            '''Transformer only supports 1D, 2D or 3D'''
        assert type(tess_size) == list or type(tess_size) == tuple, \
            '''Argument tess_size must be a list or tuple'''
        assert all([type(e)==int for e in tess_size]), \
            '''All elements of tess_size must be integers'''
        assert all([e > 0 for e in tess_size]), \
            '''All elements of tess_size must be positive'''
        assert backend in ['numpy', 'tensorflow', 'pytorch'], \
            '''Unknown backend, choose between 'numpy', 'tensorflow' or 'pytorch' '''
        assert device in ['cpu', 'gpu'], \
            '''Unknown device, choose between 'cpu' or 'gpu' '''
        if backend == 'numpy':
            assert device == 'cpu', '''Cannot use gpu with numpy backend '''
        assert type(zero_boundary) == bool, \
            '''Argument zero_boundary must be True or False'''
        assert type(volume_perservation) == bool, \
            '''Argument volume_perservation must be True or False'''
        assert type(override) == bool, \
            '''Argument override must be True or False '''
            
    #%%
    def _check_type(self, x):
        """ Assert that the type of x is compatible with the class i.e
                numpy backend expects np.array
                pytorch backend expects torch.tensor
                tensorflow backend expects tf.tensor
        """
        assert isinstance(x, self.backend.backend_type()), \
            ''' Input has type {0} but expected type {1} '''.format(
            type(x), self.backend.backend_type())
            
    #%%
    def _check_device(self, x):
        """ Asssert that x is on the same device (cpu or gpu) as the class """
        assert self.backend.check_device(x, self.device), '''Input is placed on 
            device {0} but the class expects it to be on device {1}'''.format(
            str(x.device), self.device)
            
    #%%
    def __repr__(self):
        output = '''
        CPAB transformer class. 
            Parameters:
                Tesselation size:           {0}
                Total number of cells:      {1}
                Theta size:                 {2}
                Domain lower bound:         {3}
                Domain upper bound:         {4}
                Zero Boundary:              {5}
                Volume perservation:        {6}
            Backend:                        {7}
        '''.format(self.params.nc, self.params.nC, self.params.d, 
            self.params.domain_min, self.params.domain_max, 
            self.params.zero_boundary, self.params.volume_perservation,
            self.backend_name)
        return output
