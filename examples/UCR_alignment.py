"""
Created on Oct  2019
Modified for torch Aug 2020

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
import torch

if module_path not in sys.path:
    sys.path.append(module_path)

# From helper
from helper.plotting_torch import plot_signals
from helper.UCR_loader import get_UCR_data

# from models
from models.train_utils import ExperimentsManager, DTAN_args
from models.train_model import train

def argparser():
    parser = argparse.ArgumentParser(description='Process args')
    parser.add_argument('--dataset', type=str, default='ECGFiveDays')
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
    parser.add_argument('--n_epochs', type=int, default=500,
                        help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="batch size")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument('--dpath', type=str, default="", help="dataset dir path, default examples/data")
    args = parser.parse_args()
    return args

def run_UCR_alignment(args, dataset_name="ECGFiveDays"):
    """
    Run an example of the full training pipline for DTAN on a UCR dataset.
    After training:
        - The model checkpoint (based on minimal validation loss) at checkpoint dir.
        - Plots alignment, within class, for train and test set.

    Args:
        args: described at argparser. args for training, CPAB basis, etc
        dataset_name: dataset dir name at examples/data

    """

    # Print args
    print(args)

    # Data
    datadir = args.dpath #"data/UCR/UCR_TS_Archive_2015"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = f"{dataset_name}_exp"
    # Plotting flag
    plot_signals_flag = True

    # Init an instance of the experiment class. Holds results
    # and trainning param such as lr, n_epochs etc
    expManager = ExperimentsManager()
    expManager.add_experiment(exp_name, n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr, device=device)
    Experiment = expManager[exp_name]

    # DTAN args
    DTANargs1 = DTAN_args(tess_size=args.tess_size,
                          smoothness_prior=args.smoothness_prior,
                          lambda_smooth=args.lambda_smooth,
                          lambda_var=args.lambda_var,
                          n_recurrences=args.n_recurrences,
                          zero_boundary=True,
                          )
    expManager[exp_name].add_DTAN_arg(DTANargs1)

    DTANargs = Experiment.get_DTAN_args()
    train_loader, validation_loader, test_loader = get_UCR_data(dataset_name=dataset_name,
                                                                datadir=datadir,
                                                                batch_size=Experiment.batch_size)


    # Train model
    model = train(train_loader, validation_loader, DTANargs, Experiment, print_model=True)

    # Plot aligned signals
    if plot_signals_flag:
        # Plot test data
        plot_signals(model, device, datadir, dataset_name)



if __name__ == "__main__":
    args = argparser()
    run_UCR_alignment(args, dataset_name=args.dataset)




# References:
# [1] - Diffeomorphic Temporal Alignment Nets (NeurIPS 2019)
