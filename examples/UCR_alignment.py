"""
Created on Oct  2019

author: ronsha

End-to-end alignment of datasets belonging to the UCR archive.
If you call 'run_UCR_alignment(...)' from another scripts, make sure to construct and pass an args class.
You can see "UCR_NCC.py" for example.

Plotting:
By default, the script will produce figures for each class in a similar fashion to figure 1. from [1].
You disable it via 'plot_signals_flag'.
In addition, you can:
1. Plot the output of RDTAN at each recurrence
2. Create an animation of RDTAN at each recurrence.

This is possible by simply uncommenting the relevant lines.
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
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help="number of epochs")
    args = parser.parse_args()
    return args

def run_UCR_alignment(args, dataset_name="ECGFiveDays"):

    # Print args
    print(args)

    # Data
    datadir = "data/"
    X_train, X_test, y_train, y_test = load_UCR_data(datadir, dataset_name)
    # Data info
    input_shape, n_classes = get_dataset_info(dataset_name, X_train, X_test, y_train, y_test, print_info=True)

    # Plotting flag
    plot_signals_flag = True

    # run network - args holds all training related parameters
    model, DTAN = run_alignment_network(X_train, y_train, args)

    # Align data - forward pass the data through the network
    # create transformer function
    DTAN_aligner = K.function(inputs=[model.input], outputs=[model.layers[-1].output])

    #X_train_aligned = np.squeeze(DTAN_aligner([X_train]))
    X_test_aligned = np.squeeze(DTAN_aligner([X_test]))

    # plot results - similar format to Figure 1 in [1]
    if plot_signals_flag:
        # Plot test data
        plot_signals(X_test, X_test_aligned, y_test, ratio=[10,6], dataset_name=dataset_name)

    ### Plot RDTAN - comment out for usage ###

    # Plot output at each recurrence
    #DTAN.plot_RDTAN_outputs(DTAN, X_train, y_train, ratio=[6,4])

    # Create animation, saves as gif in this script's dir
    #RDTAN_animation(DTAN, X_test, y_test, args.n_recurrences, args)

if __name__ == "__main__":
    args = argparser()
    run_UCR_alignment(args)




# References:
# [1] - Diffeomorphic Temporal Alignment Nets (NeurIPS 2019)