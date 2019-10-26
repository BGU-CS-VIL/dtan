"""
Created on Oct  2019

author: ronsha
"""

# From tensorflow
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python import keras

# From cpablib
from libcpab.tensorflow import cpab

# From local files
from models.get_locnet import get_locnet
from DTAN.DTAN_layer import DTANLayer
from DTAN.alignmnet_loss import alignment_loss
from DTAN.smoothness_prior import smoothness_norm

def get_transformer_module(inputs, n_stack, T, d):
    '''

    :param inputs: Keras input layer
    :param n_stack: number of transformer to stack
    :param T: cpab transformer
    :param d: theta dim
    :return: stacked transformer module
    '''


    locnets = []
    DTANS = []
    for i in range(n_stack):
        transformer_name = f"Temporal_Alignment_Layer{i}"

        # shared weights - Create locnet only once
        if i==0:
            locnet = get_locnet(inputs, output_shape=d)
        DTANS.append(DTANLayer(T, locnet, name=transformer_name))


    # connect transformers
    for i in range(n_stack):
        inputs = DTANS[i](inputs)

    outputs = inputs

    return outputs, DTANS, locnets


def run_alignment_network(X_train ,y_train, args):

    #  Needed here for some reason
    from helper.keras_callbacks import best_val_loss_model

    # General
    input_shape = X_train.shape[1:]
    # Create cpab transformer
    T = cpab(tess_size=[args.tess_size, ], return_tf_tensors=True, zero_boundary=args.zero_boundary)
    d = T.get_theta_dim()


    # Construct localization network
    inputs = keras.Input(shape=input_shape)

    # get (Recurrent) DTAN
    DTAN_output, DTANs, locnets = get_transformer_module(inputs, args.n_recurrences, T, d)
    model_output = DTAN_output
    # Build Model
    model = Model(inputs=inputs, outputs=model_output)


    # Add smoothness prior on theta for each recurrence
    if args.smoothness_prior:
        for i in range(args.n_recurrences):
            if i == 0:
                theta = DTANs[i].get_theta(inputs)
            else:
                theta = DTANs[i].get_theta(DTANs[i-1](inputs))

            model.add_loss(smoothness_norm(T, tf.squeeze(theta), args.lambda_smooth, args.lambda_var))

    # print model summary
    model.summary()

    # The compile step specifies the training configuration.
    opt = tf.keras.optimizers.Adam(lr=0.0001)
    # Add alignment loss
    model.compile(loss=alignment_loss(model_output, loss_type ="l2"), optimizer=opt)

    # Callbacks
    callbacks = []
    # best validation loss
    best_val_loss = True
    if best_val_loss:
        best_val_loss = best_val_loss_model(min_epoch=0)
        callbacks.append(best_val_loss)

    history = model.fit(X_train ,y_train,
                        validation_split=0.2,
                        batch_size=64,
                        epochs=1000,
                        callbacks = callbacks,
                        verbose=1)

    return model
