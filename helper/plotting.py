"""
Created on Oct  2019

author: ronsha
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_within_class(X, ratio):
    '''
    Simple plotting of the mean signal and 10 random samples within a single class.
    :param X: single class time-series data. numpy array - (signals, time-steps, channels)
    :param ratio: figure ratio
    :return: None
    '''

    [w,h] = ratio
    fig = plt.figure(figsize=(w,h))

    # Sample
    n_signals = X.shape[0]  # number of samples within class
    indices = np.random.choice(n_signals, 10)
    X_samples = np.squeeze(X[indices,:,:])
    X_mean = np.mean(X_samples, axis=0)

    # Plot
    plt.plot(np.squeeze(X_samples.T), color='grey', alpha=0.2)
    plt.plot(np.squeeze(X_mean), alpha=0.8)
    plt.show()

def plot_mean_signal(X_aligned_within_class, X_within_class, ratio, class_num, dataset_name):

    #check data dim
    if len(X_aligned_within_class.shape) < 3:
        X_aligned_within_class = np.expand_dims(X_aligned_within_class, axis=-1)

    # data dims: (number of samples, dim, channels)
    n_signals = len(X_within_class)  # number of samples within class
    N = 10

    # Sample random signals
    input_shape = X_within_class.shape[1:]  # (dims, channels)
    signal_len = input_shape[0]
    n_channels = input_shape[-1]

    indices = np.random.choice(n_signals, N)  # N samples
    X_within_class = X_within_class[indices, :, :]  # get N samples, all channels
    X_aligned_within_class = X_aligned_within_class[indices, :, :]

    # Compute mean signal and variance
    X_mean_t = np.mean(X_aligned_within_class, axis=0)
    X_std_t = np.std(X_aligned_within_class, axis=0)
    upper_t = X_mean_t + X_std_t
    lower_t = X_mean_t - X_std_t

    X_mean = np.mean(X_within_class, axis=0)
    X_std = np.std(X_within_class, axis=0)
    upper = X_mean + X_std
    lower = X_mean - X_std

    # set figure size
    [w, h] = ratio  # width, height
    f = plt.figure(1)
    plt.style.use('seaborn-darkgrid')
    f.set_size_inches(w, n_channels * h)

    title_font = 18
    rows = 2
    cols = 2
    plot_idx = 1
    # plot each channel
    for channel in range(n_channels):
        t = range(input_shape[0])
        # Misaligned Signals
        ax1 = f.add_subplot(rows, cols, plot_idx)
        ax1.plot(X_within_class[:, :, channel].T)
        plt.tight_layout()
        plt.xlim(0, signal_len)

        if n_channels == 1:
            #plt.title("%d random test samples" % (N))
            plt.title("Misaligned signals", fontsize=title_font)
        else:
            plt.title("Channel: %d, %d random test samples" % (channel, N))
        plot_idx += 1

        # Misaligned Mean
        ax2 = f.add_subplot(rows, cols, plot_idx)
        ax2.plot(t, X_mean[:, channel], 'r',label='Average signal')
        ax2.fill_between(t, upper[:, channel], lower[:, channel], color='r', alpha=0.2, label=r"$\pm\sigma$")
        #plt.legend(loc='upper right', fontsize=12, frameon=True)
        plt.xlim(0, signal_len)

        if n_channels ==1:
            plt.title("Misaligned average signal", fontsize=title_font)
        else:
            plt.title("Channel: %d, Test data mean signal (%d samples)" % (channel,N))

        plot_idx += 1


        # Aligned signals
        ax3 = f.add_subplot(rows, cols, plot_idx)
        ax3.plot(X_aligned_within_class[:, :, channel].T)
        plt.title("DTAN aligned signals", fontsize=title_font)
        plt.xlim(0, signal_len)

        plot_idx += 1

        # Aligned Mean
        ax4 = f.add_subplot(rows, cols, plot_idx)
        # plot transformed signal
        ax4.plot(t, X_mean_t[:, channel], label='Average signal')
        ax4.fill_between(t, upper_t[:, channel], lower_t[:, channel], color='#539caf', alpha=0.6, label=r"$\pm\sigma$")


        #plt.legend(loc='upper right', fontsize=12, frameon=True)
        plt.title("DTAN average signal", fontsize=title_font)
        plt.xlim(0, signal_len)
        plt.tight_layout()

        plot_idx += 1

    #plt.savefig(f'{dataset_name}_{int(class_num)}.pdf', format='pdf')
    plt.tight_layout()
    plt.suptitle(f"{dataset_name}: class-{class_num}")
    plt.show()


def plot_signals(X, X_aligned, y, ratio=[6,4], dataset_name=""):
    '''

    :param X: original test data [n_sample,timesteps, channels]
    :param X_aligned: aligned test data [n_sample,timesteps, channels]
    :param y_n: labels in numeric format
    :param fig_dir: output dir for figures
    :param dataset_name: string
    :param save_figs: boolean.
    :param barycenters: bool. Compute and compare to barycenter in figure.
    :return: plot before and after alignment mean signal and random samples.
    '''



    counter = 0
    class_names = np.unique(y, axis=0)
    for class_num in class_names:
        print('\n\nclass num for plotting:', counter)
        counter += 1
        class_idx = y == class_num
        X_within_class = X[class_idx]
        X_aligned_within_class = X_aligned[class_idx]

        # plot aligned and misaligned mean signal
        plot_mean_signal(X_aligned_within_class, X_within_class, ratio, class_num, dataset_name)

