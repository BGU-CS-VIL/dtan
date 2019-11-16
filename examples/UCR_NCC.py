"""
Created on Nov  2019

author: ronsha


Reproduction pipeline for the NCC experiment in [1]
using the same hyper-parameters and training scheme
"""
import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


# standard
import numpy as np
# From tensorflow
from tensorflow.keras import backend as K
# Local dirs
from helper.results_loader import get_DTAN_NCC_HP
from helper.UCR_loader import load_UCR_data
from helper.NCC import NearestCentroidClassification
# models
from models.train_model import run_alignment_network
from models.args import args

if  __name__ == "__main__":

    dataset_name = "ECGFiveDays"
    # Get hyper-parameters used in [1] from the pickle file
    lambda_smooth, lambda_var, n_recurrences = get_DTAN_NCC_HP(dataset_name)
    # Construct args class
    args = args(lambda_smooth=lambda_smooth, lambda_var=lambda_var, n_recurrences=n_recurrences)
    args.n_epochs = 2000
    # Print args
    print(args)

    # Data
    datadir = "data/"
    X_train, X_test, y_train, y_test = load_UCR_data(datadir, dataset_name)

    # Run network - args holds all training related parameters
    model, DTAN = run_alignment_network(X_train, y_train, args)

    # Align data - forward pass the data through the network
    # create transformer function
    DTAN_aligner = K.function(inputs=[model.input], outputs=[model.layers[-1].output])

    X_train_aligned = DTAN_aligner([X_train])
    X_test_aligned = DTAN_aligner([X_test])
    X_train_aligned = np.asarray(X_train_aligned[0]) # Convert from list numpy array
    X_test_aligned = np.asarray(X_test_aligned[0])

    ### NCC ##
    NearestCentroidClassification(X_train_aligned, X_test_aligned, y_train, y_test, dataset_name)

# References:
# [1] - Diffeomorphic Temporal Alignment Nets (NeurIPS 2019)