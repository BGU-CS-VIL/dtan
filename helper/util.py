"""
Created on Oct  2019

author: ronsha
"""

from scipy.signal import resample
from contextlib import redirect_stdout
import numpy as np
from scipy import signal
import json
from tensorflow.python.keras.utils import to_categorical


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


def cpab_summary(args):
    '''
    Ags:
        args: CPAB args
    Returns:
        string containing cpab args separated by a new line

    '''
    print("CPAB Summary:")

    # Create model description for figure
    textstr = ""
    for key, value in args.__dict__.items():
        if not key.startswith('__'):
            textstr += key + ": " + str(value) + "\n"
    return textstr


def save_model_summary(model, path, name):
    with open(f'{path}/{name}_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()


def print_classification_results(results):
    for dataset, dataset_results in results.items():
        print('\n', dataset)
        if dataset != "args":
            for model_key, acc_type in results[dataset].items():
                print('\n', model_key, acc_type)


def print_NCC_results(results):
    for dataset, dataset_results in results.items():
        print('\n', dataset)
        if dataset !="args":
            for model_key in results[dataset].keys():
                print('\n', model_key)
                if isinstance(results[dataset][model_key], dict):
                    print("[Mean accuracy, mean std]")
                    for method, acc in results[dataset][model_key].items():
                        print(method, acc)
                else:
                    print(results[dataset][model_key])


def get_test_results(val_results, test_results):
    # delete values for bookkeeping
    if "args" in val_results:
        del val_results["args"]
    if "args" in test_results:
        del test_results["args"]

    best_val_model = {}
    final_test_results = {}

    for dataset, dataset_results in val_results.items():
        best_val_model[dataset] = ""
        final_test_results[dataset] = {}
        curr_acc = 0
        for model_key in val_results[dataset].keys():
            if val_results[dataset][model_key]["DTAN-EUC"][0] > curr_acc:
                curr_acc = val_results[dataset][model_key]["DTAN-EUC"][0]
                best_val_model[dataset] = model_key

        # Get test results for best validation model
        best_model = best_val_model[dataset]

        final_test_results[dataset] = {}
        final_test_results[dataset][best_model] = test_results[dataset][best_model]["DTAN-EUC"]
    return final_test_results



def from_json(fdir, fname):
    with open(f'{fdir}/{fname}.json') as json_file:
        data = json.load(json_file)
        return data


# resample X
def resample_X(X, sample_rate):
    n_samples = X.shape[0]  # number of samples
    channels = X[0].shape[-1]
    X_resampled = np.zeros((n_samples, sample_rate, channels))

    for i in range(n_samples):
        X_resampled[i] = resample(X[i], sample_rate)
    return X_resampled

def numerical_to_categorical(y_train_n, y_val_n, y_test_n):
    #n_classes = len(np.unique(y_train_n))
    y_train = to_categorical(y_train_n)
    y_val = to_categorical(y_val_n)
    y_test = to_categorical(y_test_n)

    return y_train, y_val, y_test


def bandpass_filter(X, lowcut, highcut, fs):

    filter1 = signal.firwin(400, cutoff=[lowcut / (fs / 2), highcut / (fs / 2)], pass_zero=False)

    X = np.asarray([signal.convolve(x, filter1, mode='same') for x in np.squeeze(X)])
    return np.expand_dims(X, -1)