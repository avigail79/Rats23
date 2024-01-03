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
from config import OPTIONS

# Define the priorLambda function
x1 = 0.15
x2 = 0.3
prior_lambda = lambda x: (x >= 0) * (x < x1) + (x >= x1) * (x < x2) * (x2 - x) / (x2 - x1)


def create_counts_dataframe(data, folder_path, stim_type):
    # remove "Duration time" and "No decision"
    data = data[(data['Rat Decison'] == 'Right') + (data['Rat Decison'] == 'Left')]

    # Count the number of times each direction appears
    direction_counts = data['HEADING_DIRECTION'].value_counts().sort_index()
    df_counts = direction_counts.reset_index()  # create new dataframe
    df_counts.columns = ['HEADING_DIRECTION', 'counts']

    # Count number of 'R' choices in each direction
    directions = data['HEADING_DIRECTION'].unique()
    r_counts = data[data['Rat Decison'] == 'Right'].groupby('HEADING_DIRECTION').size()
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

        result = psignifit(pfit_input, OPTIONS)
        OPTIONS['priors'] = copy.deepcopy(result['options']['priors'])
        OPTIONS['priors'][2] = prior_lambda
        result = psignifit(pfit_input, OPTIONS)

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

    plt.savefig(os.path.join(folder_path, f'pfit {date} {name}.png'))     # Save the figure
    plt.savefig(os.path.join(all_plots_path, f'pfit {date} {name}.png'))     # save to 'all plots'
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
    pfit_result = create_counts_dataframe(filtered_data, folder_path, stim_type)

    # pfit
    result = psignifit(pfit_result, OPTIONS)
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
            stim_result_path = os.path.join(results_path,
                                            f'{dates_within_range[0]} - {dates_within_range[-1]} {rat_name} results stim {s}.csv')
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
        pfit_result = pd.read_csv(os.path.join(results_path,
                                               f'{dates_within_range[0]} - {dates_within_range[-1]} {rat_name} results stim {s}.csv'))
        pfit_result = pfit_result[['HEADING_DIRECTION', 'R_counts', 'counts']]

        # pfit
        result = psignifit(pfit_result, OPTIONS)
        OPTIONS['priors'] = copy.deepcopy(result['options']['priors'])
        OPTIONS['priors'][2] = prior_lambda
        result = psignifit(pfit_result, OPTIONS)
        plot_ax = plotPsych(result)

        theta = getStandardParameters.getStandardParameters(result)
        mu = theta[0]
        sigma = theta[1]
        lambda_ = result['Fit'][2]
        gamma = result['Fit'][3]
        plot_ax.set_title(f'Bias: {mu:.2f}, Threshold: {sigma:.2f}')
        x = np.linspace(-60, 60, 500)
        sigmoid = lambda x, mu, sigma, lambda_, gamma: (
                gamma + (1 - lambda_ - gamma) * (1 / (1 + np.exp(-(x - mu) / sigma))))
        sigmoid = lambda x, mu, sigma, lambda_, gamma: 1 / (1 + np.exp(-(x - mu) / sigma))
        sigmoid_values = sigmoid(x, mu, sigma, lambda_, gamma)
        plt.plot(x, sigmoid_values, color='red', label='Sigmoid Fit')
        plt.savefig(os.path.join(results_path,
                                 f'{dates_within_range[0]} - {dates_within_range[-1]} {rat_name} results stim {s}.png'))