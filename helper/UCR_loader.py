"""
Created on Oct  2019

author: ronsha
"""


import numpy as np
import os


def load_txt_file(datadir, dataset):
    # load UCR text format

    assert os.path.isdir(datadir+"/"+dataset), f"{dataset} could not be found in {datadir}"
    fdir = datadir + '/' + dataset + '/' + dataset

    data_train = np.loadtxt(fdir+'_TRAIN',delimiter=',')
    data_test_val = np.loadtxt(fdir+'_TEST',delimiter=',')

    # get data
    X_train = data_train[:,1:]
    X_test = data_test_val[:,1:]
    # get labels (numerical, not one-hot encoded)
    y_train = data_train[:,0]
    y_test = data_test_val[:,0]

    return X_train, X_test, y_train, y_test

def load_UCR_data(datadir, dataset):
    # load data in numpy format
    X_train, X_test, y_train, y_test = load_txt_file(datadir, dataset)
    # add a thrid channel for univariate data
    if len(X_train.shape) < 3:
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)

    # Fix labels (some UCR datasets have negative labels)
    class_names = np.unique(y_train, axis=0)
    y_train_tmp = np.zeros(len(y_train))
    y_test_tmp = np.zeros(len(y_test))
    for i, class_name in enumerate(class_names):
        y_train_tmp[y_train == class_name] = i
        y_test_tmp[y_test == class_name] = i

    # Fixed
    y_train = y_train_tmp
    y_test = y_test_tmp

    return X_train, X_test, y_train, y_test