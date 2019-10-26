#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 2019

@author: ronsha
"""
from tensorflow.python.keras.engine.base_layer import Layer



#%% 
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