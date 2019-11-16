"""
Created on Oct  2019

author: ronsha
"""

# From tensorflow
import tensorflow as tf
from tensorflow.python import keras

# From cpablib

# From local files
from DTAN.DTAN_layer import DTAN_model
from DTAN.alignmnet_loss import alignment_loss


def run_alignment_network(X_train ,y_train, args):

    #  Needed here for some reason
    from helper.keras_callbacks import best_val_loss_model

    # General
    input_shape = X_train.shape[1:]



    # Construct localization network
    inputs = keras.Input(shape=input_shape)

    DTAN = DTAN_model(inputs, args.tess_size, args.smoothness_prior, args.lambda_smooth, args.lambda_var,
                 args.n_recurrences, args.zero_boundary)

    # Get Keras model
    model = DTAN.get_keras_model()
    # Get model output for alignment loss
    model_output = DTAN.get_model_output_layer()
    # print model summary
    print("##### DTAN model summary: #####")
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
                        validation_split=0.1,
                        batch_size=64,
                        epochs=args.n_epochs,
                        callbacks = callbacks,
                        verbose=1)


    return model, DTAN
