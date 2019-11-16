"""
Created on Oct  2019

author: ronsha
"""

import numpy as np



def get_dataset_info(dataset_name, X_train, X_test, y_train, y_test, print_info=True):
    N, input_shape = X_train.shape[:2]
    input_shape = X_train.shape[1:]
    n_classes = len(np.unique(y_train))

    if print_info:
        print(f"{dataset_name} dataset details:")
        print('    X train.shape:', X_train.shape)
        print('    X test.shape:', X_test.shape)
        print('    y train.shape:', y_train.shape)
        print('    y test.shape:', y_test.shape)
        print('    number of classes:', n_classes)
        print('    number of samples:', N)
        print('    data sample dim:', input_shape)

    return input_shape, n_classes


def print_model_details(locnet, model):
    print("Localization Network Summary:")
    print(locnet.summary())
    print("\nFull Network Summary:")
    print(model.summary())
