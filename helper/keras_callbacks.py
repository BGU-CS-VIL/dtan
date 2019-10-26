"""
Created on Oct  2019

author: ronsha
"""

from tensorflow.python import keras
import numpy as np

class best_val_acc_model(keras.callbacks.Callback):


    def __init__(self, min_epoch=0):
        self.min_epoch = min_epoch
        self.best_epoch = 0

    def on_train_begin(self, logs={}):
        self.best_model = keras.models.clone_model(self.model)  # according to val_acc

        self.best_val_acc = 0.0
        return

    def on_train_end(self, logs={}):
        print("returning best model with %0.3f validation accuracy (epoch %d)" % (self.best_val_acc, self.best_epoch))

        # set weights for best validation accuracy model:
        self.model.set_weights(self.best_model.get_weights())
        return

    def on_epoch_end(self, epoch, logs={}):
        curr_val_acc = logs.get('val_acc')
        if curr_val_acc > self.best_val_acc and epoch > self.min_epoch:
            self.best_model.set_weights(self.model.get_weights())
            self.best_val_acc = curr_val_acc
            self.best_epoch = epoch
        return


class best_val_loss_model(keras.callbacks.Callback):


    def __init__(self, min_epoch=0):
        self.min_epoch = min_epoch
        self.best_epoch = 0

    def on_train_begin(self, logs={}):
        self.best_model = keras.models.clone_model(self.model)  # according to val_loss

        self.best_val_loss = np.inf
        return

    def on_train_end(self, logs={}):
        print("Returning best model with %0.3f validation loss (epoch %d)" % (self.best_val_loss, self.best_epoch))

        # set weights for best validation accuracy model:
        self.model.set_weights(self.best_model.get_weights())
        return

    def on_epoch_end(self, epoch, logs={}):
        curr_val_loss = logs.get('val_loss')
        if curr_val_loss < self.best_val_loss and epoch > self.min_epoch:
            self.best_model.set_weights(self.model.get_weights())
            self.best_val_loss = curr_val_loss
            self.best_epoch = epoch
        return