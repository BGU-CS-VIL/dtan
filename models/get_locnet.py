"""
Created on Oct  2019

author: ronsha
"""

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Activation, BatchNormalization, Conv1D, Dropout, MaxPool1D, Flatten

# Localization Network template:
def get_locnet(locnet_inputs, output_shape):
    # Define filters
    c1 = 128
    c2 = 64
    c3 = 64

    # CNN
    inputs = locnet_inputs
    x1 = Conv1D(c1, (8), padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPool1D(2)(x1)
    x1 = Conv1D(c2, (5), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPool1D(2)(x1)
    x1 = Conv1D(c3, (3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPool1D(2)(x1)


    x1 = Flatten()(x1)
    # Output Layer
    locnet_outputs = Dense(output_shape, activation='tanh')(x1)

    # Define model
    locnet = Model(inputs=locnet_inputs, outputs=locnet_outputs, name="Localization Network")

    return locnet

