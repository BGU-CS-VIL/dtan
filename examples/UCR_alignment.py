"""
Created on Oct  2019

author: ronsha
"""

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
import argparse

if module_path not in sys.path:
    sys.path.append(module_path)

# From other libraries
import numpy as np

# From tensorflow
from tensorflow.keras import backend as K

# From helper
from helper.util import get_dataset_info
from helper.plotting import plot_signals, RDTAN_animation
from helper.UCR_loader import load_UCR_data
# models
from models.train_model import run_alignment_network


def argparser():
    parser = argparse.ArgumentParser(description='Process args')
    parser.add_argument('--tess_size', type=int, default=16,
                        help="CPA velocity field partition")
    parser.add_argument('--smoothness_prior', default=True,
                        help="smoothness prior flag", action='store_true')
    parser.add_argument('--no_smoothness_prior', dest='smoothness_prior', default=True,
                        help="no smoothness prior flag", action='store_false')
    parser.add_argument('--lambda_smooth', type=float, default=1,
                        help="lambda_smooth, larger values -> smoother warps")
    parser.add_argument('--lambda_var', type=float, default=0.1,
                        help="lambda_var, larger values -> larger warps")
    parser.add_argument('--n_recurrences', type=int, default=1,
                        help="number of recurrences of R-DTAN")
    parser.add_argument('--zero_boundary', type=bool, default=True,
                        help="zero boundary constrain")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argparser()
    # data
    datadir = "data/"
    dataset_name = "ECGFiveDays"

    X_train, X_test, y_train, y_test = load_UCR_data(datadir, dataset_name)
    # Data info
    input_shape, n_classes = get_dataset_info(dataset_name, X_train, X_test, y_train, y_test, print_info=True)

    # args
    plot_signals_flag = False
    # run network - args holds all training related parameters
    DTAN = run_alignment_network(X_train, y_train, args)

    # Align data - forward pass the data through the network
    # create transformer function
    DTAN_aligner = K.function(inputs=[DTAN.input], outputs=[DTAN.layers[-1].output])

    X_train_aligned = np.squeeze(DTAN_aligner([X_train]))
    X_test_aligned = np.squeeze(DTAN_aligner([X_test]))


    # plot output at each recurrence.
    # model: *trained* DTAN model

    #DTAN.plot_RDTAN_outputs(DTAN, X_train, y_train, ratio=[6,4])
    # Create animation, saves as gif
    #RDTAN_animation(DTAN, X_test, y_test, args.n_recurrences)

    # plot results - similar format to Figure 1 in [1]
    if plot_signals_flag:
        # Plot test data
        plot_signals(X_test, X_test_aligned, y_test, ratio=[10,6], dataset_name=dataset_name)

    # References:
    # [1] - Diffeomorphic Temporal Alignment Nets (NeurIPS 2013)