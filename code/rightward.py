import pandas as pd
from psignifit.psigniplot import plotPsych
from psignifit.psignifit import psignifit
from psignifit import getStandardParameters
import matplotlib.pyplot as plt
import psignifit as ps
import matplotlib.gridspec as gridspec
import numpy as np
import os
import copy
from utils import is_date_within_range
from math import sqrt


# Define the priorLambda function
x1 = 0.15
x2 = 0.3
prior_lambda = lambda x: (x >= 0) * (x < x1) + (x >= x1) * (x < x2) * (x2 - x) / (x2 - x1)


def sigmoid_plot(x, mu=0, sigma=1, L=0.01):
    """
    Compute the sigmoid function with given parameters.
    :param x: Input values (numpy array).
    :param bias: Shifts the curve left or right.
    :param width: Determines the steepness of the curve.
    :param lapse_rate: Represents the deviation from the min and max values.
    :return: Sigmoid values corresponding to x.
    """
    # y = gamma + (1 - gamma - lambda_) / (1 + np.exp(-(x - mu) / sigma))
    y = L / (1 + np.exp(-sigma * (x - mu)))
    return plt.plot(x, y, label='Sigmoid Function')


def create_counts_dataframe(curr_data, folder_path, stim_type):
    # todo: remove no decision

    # Count the number of times each direction appears
    direction_counts = curr_data['HEADING_DIRECTION'].value_counts().sort_index()
    df_counts = direction_counts.reset_index()  # create new dataframe
    df_counts.columns = ['HEADING_DIRECTION', 'counts']

    # Count number of 'R' choices in each direction
    directions = curr_data['HEADING_DIRECTION'].unique()
    r_counts = curr_data[curr_data['Rat Decison'] == 'Right'].groupby('HEADING_DIRECTION').size()
    r_counts = r_counts.reindex(directions, fill_value=0).sort_index()
    df_r_counts = r_counts.reset_index()
    df_r_counts.columns = ['HEADING_DIRECTION', 'R_counts']

    # Merge df_counts and df_r_counts
    pfit_inputs = pd.merge(df_r_counts, df_counts, on='HEADING_DIRECTION', how='outer')

    # Apply the calculation for negative HEADING_DIRECTION
    # mask = pfit_inputs['HEADING_DIRECTION'] < 0
    # pfit_inputs.loc[mask, 'R_counts'] = pfit_inputs.loc[mask, 'counts'] - pfit_inputs.loc[mask, 'R_counts']

    file_path = os.path.join(folder_path, f'pfit_inputs_stim{stim_type}.csv')
    pfit_inputs.to_csv(file_path)
    return pfit_inputs


def create_pfit_multiple_stim(stim_types, data, folder_path):
    # Create a figure with subplots
    num_stim_types = len(stim_types)
    fig, axes = plt.subplots(1, num_stim_types, figsize=(10 * num_stim_types, 8))
    all_plots_path = os.path.join(folder_path[:-11], 'all plots')

    if num_stim_types == 1:
        axes = [axes]

    date = folder_path.split('\\')[-1]
    name = folder_path.split('\\')[-2][5:]
    fig.suptitle(f'{date} {name}, StimTypes: {stim_types}', fontsize=16)

    for i, stim_type in enumerate(stim_types):
        # Filter data for the current stimulus type
        filtered_data = data[data['STIMULUS_TYPE'] == stim_type]
        pfit_input = create_counts_dataframe(filtered_data, folder_path, stim_type)

        options = {
            'sigmoidName': 'norm',
            'expType': 'equalAsymptote'
            # 'stimulusRange': [-90, 90]
        }
        result = psignifit(pfit_input, options)
        # options['priors'] = copy.deepcopy(result['options']['priors'])
        # options['priors'][2] = prior_lambda
        # result = psignifit(pfit_input, options)

        plotPsych(result, axisHandle=axes[i])
        theta = getStandardParameters.getStandardParameters(result)
        mu = theta[0]
        sigma = theta[1]
        axes[i].set_title(f'StimType: {stim_type}, Bias: {mu:.2f}, Threshold: {sigma:.2f}')

        all_pfit_path = os.path.join(all_plots_path, f'all pfit results stim {stim_type}.csv')
        if os.path.isfile(all_pfit_path):
            all_pfit_results = pd.read_csv(all_pfit_path)
            all_pfit_results = all_pfit_results[['HEADING_DIRECTION', 'R_counts', 'counts']]
            all_pfit_results['R_counts'] += pfit_input['R_counts']
            all_pfit_results['counts'] += pfit_input['counts']
            all_pfit_results.to_csv(all_pfit_path)
        else:
            pfit_input.to_csv(all_pfit_path)

    # Save the figure
    plt.savefig(os.path.join(folder_path, f'pfit {date} {name}.png'))
    # save to 'all plots'
    plt.savefig(os.path.join(all_plots_path, f'pfit {date} {name}.png'))
    return


def create_pfit_one_stim(data, folder_path):
    stim_type = [2]
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    date = folder_path.split('\\')[-1]
    name = folder_path.split('\\')[-2][5:]
    fig.suptitle(f'{date} {name}, StimType:{stim_type}', fontsize=16)

    # Filter data for the current stimulus type
    filtered_data = data[data['STIMULUS_TYPE'] == stim_type[0]]
    df_counts = create_counts_dataframe(filtered_data, folder_path, stim_type)

    # pfit
    options = {
        'sigmoidName': 'norm',
        'expType': 'equalAsymptote',
        # 'priors': {3: prior_lambda},
        'stimulusRange': [-90, 90]
    }
    result = psignifit(df_counts, options)
    plot_ax = plotPsych(result)

    theta = getStandardParameters.getStandardParameters(result)
    mu = theta[0]
    sigma = theta[1]
    plot_ax.set_title(f' Bias: {mu:.2f}, Threshold: {sigma:.2f}')

    date = folder_path.split('\\')[-1]
    rat_name = folder_path.split('\\')[-2][5:]
    plt.savefig(os.path.join(folder_path, f'pfit {date} {rat_name}.png'))
    # plt.show()

    # save to 'all plots'
    all_plots_path = os.path.join(folder_path[:-11], 'all plots')
    plt.savefig(os.path.join(all_plots_path, f'pfit {date} {rat_name}.png'))
    # all_pfit_path = os.path.join(all_plots_path, 'all pfit results.csv')
    # if os.path.isfile(all_pfit_path):
    #     all_pfit_results = pd.read_csv(all_pfit_path)
    #     all_pfit_results = all_pfit_results[['HEADING_DIRECTION', 'R_counts', 'counts']]
    #     all_pfit_results['R_counts'] += df_counts['R_counts']
    #     all_pfit_results['counts'] += df_counts['counts']
    #     all_pfit_results.to_csv(all_pfit_path)
    # else:
    #     df_counts.to_csv(all_pfit_path)

    return


def create_group_pfit(folder_path):
    listdir = os.listdir(folder_path)
    stim_types = [file[22:-4] for file in listdir if file.startswith('all pfit results stim')]


    for s in stim_types:
        fig = plt.figure(figsize=(10, 6))
        name = folder_path.split('\\')[-2][5:]
        fig.suptitle(f'{name} (all results), StimType:{s}', fontsize=16)

        all_pfit_results = pd.read_csv(os.path.join(folder_path, f'all pfit results stim {s}.csv'))
        all_pfit_results = all_pfit_results[['HEADING_DIRECTION', 'R_counts', 'counts']]

        # pfit
        options = {
            'sigmoidName': 'norm',
            'expType': 'equalAsymptote',
            # 'priors': {3: prior_lambda},
            'stimulusRange': [-90, 90]
        }
        result = psignifit(all_pfit_results, options)
        plot_ax = plotPsych(result)

        theta = getStandardParameters.getStandardParameters(result)
        mu = theta[0]
        sigma = theta[1]
        plot_ax.set_title(f' Bias: {mu:.2f}, Threshold: {sigma:.2f}')

        date = folder_path.split('\\')[-1]
        rat_name = folder_path.split('\\')[-2][5:]
        plt.savefig(os.path.join(folder_path, f'all results {rat_name} stim{s}.png'))
    return


def sigmoid(x, mu, sigma, L):
    return L / (1 + np.exp(-(x-mu) / sigma))


def create_pooled_results(rat_folder_path, start_date, end_date, stim_types):
    listdir = os.listdir(rat_folder_path)[:-2]
    dates_within_range = [date for date in listdir if is_date_within_range(date, start_date, end_date)]
    rat_name = rat_folder_path.split('\\')[-1][5:]
    results_path = os.path.join(rat_folder_path, 'results')

    for date in dates_within_range:
        path1 = os.path.join(rat_folder_path, date)
        for s in stim_types:
            path2 = os.path.join(path1, f'pfit_inputs_stim{s}.csv')
            pfit_res = pd.read_csv(path2)
            stim_result_path = os.path.join(results_path, f'{dates_within_range[0]} - {dates_within_range[-1]} {rat_name} results stim {s}.csv')
            if os.path.isfile(stim_result_path):
                res = pd.read_csv(stim_result_path)
                res = res[['HEADING_DIRECTION', 'R_counts', 'counts']]
                res['R_counts'] += pfit_res['R_counts']
                res['counts'] += pfit_res['counts']
                res.to_csv(stim_result_path)
            else:
                pfit_res.to_csv(stim_result_path)

    for s in stim_types:
        fig = plt.figure(figsize=(10, 6))
        fig.suptitle(f'{rat_name} {dates_within_range[0]}-{dates_within_range[-1]}, StimType:{s}', fontsize=16)
        pfit_result = pd.read_csv(os.path.join(results_path, f'{dates_within_range[0]} - {dates_within_range[-1]} {rat_name} results stim {s}.csv'))
        pfit_result = pfit_result[['HEADING_DIRECTION', 'R_counts', 'counts']]

        # pfit
        options = {
            'sigmoidName': 'norm',
            'expType': 'equalAsymptote'
            # 'stimulusRange': [-90, 90]
        }
        result = psignifit(pfit_result, options)
        # options['priors'] = copy.deepcopy(result['options']['priors'])
        # options['priors'][2] = prior_lambda
        # result = psignifit(pfit_result, options)
        plot_ax = plotPsych(result)

        theta = getStandardParameters.getStandardParameters(result)
        mu = theta[0]
        sigma = theta[1]
        lambda_ = result['Fit'][2]
        gamma = result['Fit'][3]
        plot_ax.set_title(f'Bias: {mu:.2f}, Threshold: {sigma:.2f}')

        # heading_directions = [i for i in stim_result_path['HEADING_DIRECTION'].astype(int)]
        # counts = np.array([i for i in stim_result_path['counts'].astype(int)])
        # R_counts = np.array([i for i in stim_result_path['R_counts'].astype(int)])
        # proportion_right = R_counts / counts
        # x_values = np.linspace(-60, 60, 500)
        # sigmoid_values = sigmoid(x_values, mu, sigma, lambda_)
        # plt.scatter(heading_directions, proportion_right, label='Experimental Data')
        # plt.plot(x_values, sigmoid_values, color='red', label='Sigmoid Fit')
        # plt.xlabel('Heading Direction')
        # plt.ylabel('Proportion Choosing Right')
        # plt.title('Sigmoid Fit to Experimental Data')
        # plt.legend()
        plt.savefig(os.path.join(results_path, f'{dates_within_range[0]} - {dates_within_range[-1]} {rat_name} results stim {s}.png'))



        # x_values = []  # Creating a new list where each heading direction is repeated according to its count
        # for direction, count in zip(heading_directions, counts):
        #     x_values.extend([direction] * count)
        # sigmoid_plot(x_values, mu, sigma, lambda_)
        # plt.savefig(os.path.join(results_path, f'{dates_within_range[0]} - {dates_within_range[-1]} {rat_name} results stim {s}.png'))


if __name__ == '__main__':
    # WORK_PATH = os.getcwd()
    # DATA_PATH = os.path.join(WORK_PATH, 'Data')
    # rats_folders = os.listdir(DATA_PATH)
    # rat_folder_path = os.path.join(DATA_PATH, '36 - Narkis')
    # start_date = '2023_11_21'
    # end_date = '2023_12_05'
    # create_group_results(rat_folder_path, start_date, end_date, [2, 10])

    p = r'C:\Zaidel\Rats\NL\Data\36 - Narkis\results\2023_11_21 - 2023_12_05 Narkis results stim 2.csv'
    stim_results = pd.read_csv(p)
    stim_results = stim_results[['HEADING_DIRECTION', 'R_counts', 'counts']]


    # pfit
    options = {
        'sigmoidName': 'norm',
        'expType': 'equalAsymptote',
        'stimulusRange': [-90, 90]
    }

    x1 = 0.2
    x2 = 0.3
    prior_lambda = lambda x: (x >= 0) * (x < x1) + (x >= x1) * (x < x2) * (x2 - x) / (x2 - x1)

    result = psignifit(stim_results, options)
    # options['priors'] = copy.deepcopy(result['options']['priors'])
    # options['priors'][2] = prior_lambda
    # result = psignifit(stim_results, options)
    plot_ax = plotPsych(result)

    theta = getStandardParameters.getStandardParameters(result)
    mu = theta[0]
    sigma = theta[1]
    lambda_ = result['Fit'][2]
    gamma = result['Fit'][3]
    plot_ax.set_title(f'mu: {mu:.3f}, sigma: {sigma:.3f}, [{lambda_:.3f}, {gamma:.3f}]')

    x = np.linspace(-60, 60, 500)
    # sigmoid = lambda x, mu, sigma, lambda_, gamma: gamma + (1 - lambda_ - gamma) * (1 / (1 + np.exp(-(x - mu) / sigma)))
    sigmoid = lambda x, mu, sigma, lambda_, gamma: 1 / (1 + np.exp(-(x - mu) / sigma))
    sigmoid_values = sigmoid(x, mu, sigma, lambda_, gamma)

    plt.plot(x, sigmoid_values, color='red', label='Sigmoid Fit')
    plt.savefig(os.path.join(r'C:\Users\user\Downloads\trial1.png'))




