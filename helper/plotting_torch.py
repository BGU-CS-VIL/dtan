import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from helper.UCR_loader import processed_UCR_data
from tslearn.datasets import UCR_UEA_datasets


def plot_mean_signal(X_aligned_within_class, X_within_class, ratio, class_num, dataset_name, N=10):

    #check data dim
    if len(X_aligned_within_class.shape) < 3:
        X_aligned_within_class = np.expand_dims(X_aligned_within_class, axis=-1)

    # data dims: (number of samples, dim, channels)
    n_signals = len(X_within_class)  # number of samples within class

    # Sample random signals
    input_shape = X_within_class.shape[1:]  # (channels, dims) PyTorch
    signal_len = input_shape[1]
    n_channels = input_shape[0]

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
    # plot each channel
    for channel in range(n_channels):
        plot_idx = 1
        t = range(input_shape[1])
        # Misaligned Signals
        if channel == 0:
            ax1 = f.add_subplot(rows, cols, plot_idx)
        ax1.plot(X_within_class[:, channel,:].T)
        plt.tight_layout()
        plt.xlim(0, signal_len)

        if n_channels == 1:
            #plt.title("%d random test samples" % (N))
            plt.title("Misaligned signals", fontsize=title_font)
        else:
            plt.title("Channel: %d, %d random test samples" % (channel, N))
        plot_idx += 1

        # Misaligned Mean
        if channel == 0:
            ax2 = f.add_subplot(rows, cols, plot_idx)
        if n_channels == 1:
            ax2.plot(t, X_mean[channel], 'r',label=f'Average signal-channel:{channel}')
            ax2.fill_between(t, upper[channel], lower[channel], color='r', alpha=0.2, label=r"$\pm\sigma$")
        else:
            ax2.plot(t, X_mean[channel,:], label=f'Average signal-channel:{channel}')

        plt.legend(loc='upper right', fontsize=12, frameon=True)
        plt.xlim(0, signal_len)

        if n_channels ==1:
            plt.title("Misaligned average signal", fontsize=title_font)
        else:
            plt.title(f"Channel: {channel}, Test data mean signal ({N} samples)")

        plot_idx += 1


        # Aligned signals
        if channel == 0:
            ax3 = f.add_subplot(rows, cols, plot_idx)
        ax3.plot(X_aligned_within_class[:, channel,:].T)
        plt.title("DTAN aligned signals", fontsize=title_font)
        plt.xlim(0, signal_len)

        plot_idx += 1

        # Aligned Mean
        if channel == 0:
            ax4 = f.add_subplot(rows, cols, plot_idx)
        # plot transformed signal
        ax4.plot(t, X_mean_t[channel,:], label=f'Average signal-channel:{channel}')
        if n_channels == 1:
            ax4.fill_between(t, upper_t[channel], lower_t[channel], color='#539caf', alpha=0.6, label=r"$\pm\sigma$")


        plt.legend(loc='upper right', fontsize=12, frameon=True)
        plt.title("DTAN average signal", fontsize=title_font)
        plt.xlim(0, signal_len)
        plt.tight_layout()
        plot_idx += 1

    #plt.savefig(f'{dataset_name}_{int(class_num)}.pdf', format='pdf')

    plt.suptitle(f"{dataset_name}: class-{class_num}", fontsize=title_font+2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_signals(model, device, datadir, dataset_name):
    # Close any remaining plots
    plt.close('all')

    with torch.no_grad():
        # Torch channels first
        X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)
        X_train, X_test, y_train, y_test = processed_UCR_data(X_train, X_test, y_train, y_test)
        data =[X_train, X_test]
        labels = [y_train, y_test]
        set_names = ["train", "test"]
        for i in range(2):
            # torch dim
            X = torch.Tensor(data[i]).to(device)
            y = labels[i]
            classes = np.unique(y)
            transformed_input_tensor, thetas = model(X, return_theta=True)

            data_numpy = X.data.cpu().numpy()
            transformed_data_numpy = transformed_input_tensor.data.cpu().numpy()

            sns.set_style("whitegrid")
            #fig, axes = plt.subplots(1,2)
            for label in classes:
                class_idx = y == label
                X_within_class = data_numpy[class_idx]
                X_aligned_within_class = transformed_data_numpy[class_idx]
                #print(X_aligned_within_class.shape, X_within_class.shape)
                plot_mean_signal(X_aligned_within_class, X_within_class, ratio=[10,6],
                                 class_num=label, dataset_name=f"{dataset_name}-{set_names[i]}")
