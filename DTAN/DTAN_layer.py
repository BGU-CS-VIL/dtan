#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 2019

@author: ronsha
"""
# Tensorflow / Keras
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.engine.base_layer import Layer
import numpy as np
# DTAN
from models.get_locnet import get_locnet
from DTAN.smoothness_prior import smoothness_norm
from helper.plot_transformer_layer import plot_all_layers
# licpab
from libcpab.tensorflow import cpab

#
import matplotlib.pyplot as plt
#%%
class DTAN_model():
    '''

    '''
    def __init__(self, inputs, tess_size, smoothness_prior=True, lambda_smooth=0.5, lambda_var=0.1,
                 n_recurrences=1, zero_boundary=True):
        '''

        :param transformer_class:
        :param localization_net:
        '''
        # Keras input layer
        self.inputs = inputs
        # Create CPAB transformer
        self.T = cpab(tess_size=[tess_size, ], return_tf_tensors=True, zero_boundary=zero_boundary)

        # thetha dim
        self.d = self.T.get_theta_dim()
        # Smoothness prior
        self.smoothness_prior = smoothness_prior
        self.lambda_var = lambda_var
        self.lambda_smooth = lambda_smooth

        # Get (optional: Recurrent) DTAN
        # DTAN output - Keras output layer -> warpped signal
        # DTANs - transformer modules
        # locnets - the same locent (shared weights) of (R-)DTAN
        self.n_recurrences = n_recurrences
        self.DTAN_output, self.DTANs, self.locnets = self.get_transformer_module(self.inputs)

        # Build Model
        self.DTAN_model = self.build_model()

        if self.smoothness_prior:
            self.add_smoothness_prior()


    def build_model(self):
        '''
        Builds Keras model
        :return:
        '''
        DTAN_model = Model(inputs=self.inputs, outputs=self.DTAN_output, name="DTAN")
        return DTAN_model

    def get_keras_model(self):
        return self.DTAN_model

    def get_model_output_layer(self):
        return self.DTAN_output

    def get_transformer_module(self, inputs):
        '''

        :param inputs: Keras input layer
        :param n_stack: number of transformer to stack
        :param T: cpab transformer
        :param d: theta dim
        :return: stacked transformer module
        '''

        locnets = []
        DTANS = []
        for i in range(self.n_recurrences):
            transformer_name = f"Temporal_Alignment_Layer{i}"

            # shared weights - Create locnet only once
            if i == 0:
                locnet = get_locnet(self.inputs, output_shape=self.d)
                print("##### Localization Network: #####")
                locnet.summary()
            DTANS.append(DTANLayer(self.T, locnet, name=transformer_name))

        # connect transformers
        for i in range(self.n_recurrences):
            inputs = DTANS[i](inputs)

        outputs = inputs

        return outputs, DTANS, locnets

    def add_smoothness_prior(self):
    # Add smoothness prior on theta for each recurrence
        for i in range(self.n_recurrences):
            if i == 0:
                theta = self.DTANs[i].get_theta(self.inputs)
            else:
                theta = self.DTANs[i].get_theta(self.DTANs[i-1](self.inputs))

            self.DTAN_model.add_loss(smoothness_norm(self.T, tf.squeeze(theta), self.lambda_smooth, self.lambda_var))

    def plot_RDTAN_outputs(self,model, X, y, ratio=[8,6], name="movie.gif"):
        '''

        :param model: trained DTAN Keras model
        :param X:
        :param p_samples: Bool. Plot samples
        :param p_mean: bool. plot mean
        :return:
        '''
        plot_all_layers(model, X, y, self.n_recurrences, ratio, name)

#    def plot_vector_field(self, X):
 #       theta = np.squeeze(self.DTANs[-1].get_theta(X))
        nb_points = 1000
        #points = self.T.uniform_meshgrid([nb_points for i in range(self.T._ndim)])
  #      self.T.visualize_vectorfield(theta, nb_points=100)
        #vector_field = self.T.calc_vectorfield(points, theta)
        #plt.plot(vector_field)
        #plt.title("Vector Field")
        #plt.show()


class DTANLayer(Layer):
    def __init__(self, transformer_class, localization_net, **kwargs): 
        self.transformer_class = transformer_class 
        self.locnet = localization_net 
        super(DTANLayer, self).__init__(**kwargs)
     
    def build(self, input_shape): 
        self.locnet.build(input_shape) 
        self._trainable_weights = self.locnet.trainable_weights 
        super(DTANLayer, self).build(input_shape)
     
    def compute_output_shape(self, input_shape): 
        return (None, input_shape[-1],1)
 
    def get_config(self): 
        config = super(DTANLayer, self).get_config()
        config['localization_net'] = self.locnet 
        config['transformer_class'] = self.transformer_class 
        return config 


    def call(self, X, mask=None):
        theta = self.locnet.call(X) 
        output = self.transformer_class.transform_data(X, theta)
        return output

    def get_theta(self, X):
        theta = self.locnet.call(X)
        return theta


#%%
if __name__ == '__main__':
    pass