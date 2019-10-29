
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow.keras import backend as K

def plot_all_layers(model, X, y, n_recurrences, ratio):
    #plt.ioff()

    plt.style.use('seaborn-darkgrid')
    class_names = np.unique(y) # assume numerical labels
    n_classes = len(class_names)
    signal_len = X.shape[1]
    N = 10 # number of signals
    original_data_counter = 1
    counter = n_classes # for layers output

    # Set figure
    [w,h] = ratio
    fig = plt.figure(figsize=(w*n_classes,h*n_recurrences+1))
    rows = n_recurrences+1
    for i in range(n_recurrences):

        # get RDTAN[i] output
        curr_layer = K.function(inputs=[model.input], outputs=[model.layers[i+1].output])

        for class_num in class_names:
            # get signals from each class and compute mean
            class_num = int(class_num)
            train_class_idx = y == class_num
            X_within_class = X[train_class_idx]
            X_mean = np.mean(X_within_class[:N], axis=0)

            X_within_class_aligned = curr_layer([X_within_class])
            X_within_class_aligned = np.expand_dims(np.asarray(X_within_class_aligned[0]),-1)
            X_mean_aligned = np.mean(X_within_class_aligned[:N], axis=0)

            # plot original data
            if i == 0:
                ax = fig.add_subplot(rows, n_classes, original_data_counter)

                ax.plot(np.squeeze(X_within_class[:N].T), color='grey', alpha=0.2)
                ax.plot(np.squeeze(X_mean.T), alpha=0.8)
                ax.set_xticklabels([])
                ax.set_xlim(0, signal_len)
                plt.title(f"Class {int(class_num)} - Original data", fontsize=16)

                #if class_num > 0:
                #    ax.set_yticklabels([])
                ax.set_yticklabels([])

                # counter for subplots
                original_data_counter+=1


            samples = np.squeeze(X_within_class_aligned[:N].T)

            # plot each layer's output
            counter +=1
            ax = fig.add_subplot(rows,n_classes,counter)

            plt.title(f"Class {int(class_num)} - RDTAN{int(i+1)}", fontsize=16)
            ax.plot(samples, color='grey', alpha=0.2)
            ax.plot(np.squeeze(X_mean_aligned.T), alpha=0.8)
            ax.set_xlim(0, signal_len)
            #plt.plot(samples_mean)
            if i < n_recurrences-1:
                ax.set_xticklabels([])
            #if class_num > 0:
            #    ax.set_yticklabels([])
            ax.set_yticklabels([])
    plt.show()


def animate_all_layers_within_class(model, X_within_class, n_recurrences, ratio=[6,4], name="movie.gif"):

    plt.style.use('seaborn-darkgrid')

    signal_len = X_within_class.shape[1]
    N = 10 # number of signals

    # Set figure
    [w,h] = ratio
    fig, ax = plt.subplots()
    #fig.figure(figsize=(w,h))
    # Compute mean signal
    X_mean = np.mean(X_within_class[:N], axis=0)

    signals = ax.plot(np.squeeze(X_within_class[:N].T), color='grey', alpha=0.2)
    mean_signal, = ax.plot(np.squeeze(X_mean.T), alpha=0.8)
    ax.set_xlim(0, signal_len)

    def init():
        for i in range(N):
            signals[0].set_ydata([np.nan] * signal_len)
        mean_signal.set_ydata([np.nan] * signal_len)
        plt.title(f"Original data", fontsize=16)

        return signals, mean_signal


    def run(i):
        if i == 0:
            for j in range(N):
                signals[j].set_ydata(X_within_class[j])
            mean_signal.set_ydata(X_mean.T)
            plt.title(f"Original data", fontsize=16)

        else:

            # Perpare data
            curr_layer = K.function(inputs=[model.input], outputs=[model.layers[i].output])
            X_within_class_aligned = curr_layer([X_within_class])
            X_within_class_aligned = np.expand_dims(np.asarray(X_within_class_aligned[0]), -1)
            X_mean_aligned = np.mean(X_within_class_aligned[:N], axis=0)

            for j in range(N):
            # Plot
                signals[j].set_ydata(np.squeeze(X_within_class_aligned[j]))
            mean_signal.set_ydata(X_mean_aligned.T)
            plt.title(f"RDTAN{int(i)}", fontsize=16)
        return signals, mean_signal



    ani = animation.FuncAnimation(
    fig, run, init_func=init, frames=n_recurrences+1, interval=2, blit=False, repeat_delay=500)
    ani.save(name, writer='imagemagick', fps=2)
    plt.show()