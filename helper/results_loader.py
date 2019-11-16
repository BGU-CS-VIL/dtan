import pandas as pd

def get_NCC_results():
    '''

    :return: pandas DF for the best test results and respective HP
    for [1] NCC experiment:
     DTAN_results_df["dataset"] = acc: x,xx, float
                                smooth: x.xxx float
                                var: x.xxx float
                                recurrences: x int
    '''
    DTAN_results_df = pd.read_pickle("../helper/results/NeurIPS_NCC_results.pkl")
    return DTAN_results_df

def get_DTAN_NCC_HP(dataset_name):
    '''

    :param dataset_name:
    :return: lambda_smooth, lambda_var, n_recurrences that were used in [1] for the NCC experiment
    '''
    DTAN_results_df = pd.read_pickle("../helper/results/NeurIPS_NCC_results.pkl")

    lambda_smooth = DTAN_results_df.loc[[dataset_name], ["lambda_smooth"]].values[0][0]
    lambda_var = DTAN_results_df.loc[[dataset_name], ["lambda_var"]].values[0][0]
    n_recurrences = DTAN_results_df.loc[[dataset_name], ["recurrences"]].values[0][0]

    return lambda_smooth, lambda_var, n_recurrences


# References:
# [1] - Diffeomorphic Temporal Alignment Nets (NeurIPS 2019)