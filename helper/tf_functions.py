"""
Created on Oct  2019

author: ronsha
"""

import tensorflow as tf
import matplotlib.pyplot as plt


"""
Functions required by the CPAB Spatial Transformer Network implemented in TensorFlow

"""


def tf_dist_mat(x):
    with tf.name_scope('tf_pdist'):
        times = tf.shape(x)  # tf.Tensor([n], shape=(1,), dtype=int32)
        x_repeat = tf.tile(x, times)  # x.shape = n. x_repeat.shape: n*n
        x_stacked = tf.reshape(x_repeat, [times[0], times[0]])  # x_stacked.shape = (n,n)
        x_dist = tf.abs(x_stacked - tf.transpose(x_stacked))

        return x_dist


def tf_plot_graphs(A):  # A is a dict
    n_plots = len(A)
    columns = 1
    rows = n_plots
    f = plt.figure(1)
    f.set_size_inches(12, 8)
    idx = 1
    for title, item in A.items():
        plt.subplot(rows, columns, idx)
        if idx % 2 == 0:
            plt.plot(item.eval())
        else:
            plt.plot(item, 'r')
        plt.title(title)
        idx += 1

    plt.subplots_adjust(hspace=0.4)  # make subplots farther from each other.
    plt.show()



if __name__ == '__main__':
    pass