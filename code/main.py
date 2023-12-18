import numpy as np
import pandas as pd
import os
from config import *
from utils import *
from rightward import *


def session_pipline(rat_folder_path, date):
    folder_path = os.path.join(rat_folder_path, date)
    listdir = os.listdir(folder_path)
    if not listdir:
        return
    pinky_data, logfile = pd.DataFrame(), pd.DataFrame()

    # convert pinky protocol to csv and mat file
    pinky_files = [file for file in listdir if file.startswith(date) and file.endswith('.txt')]
    # Process the first pinky_files if the list is existed
    if pinky_files:
        pinky_data = pinky_process(pinky_files, folder_path)
        # create pfit
    # create_pfit_stim(pinky_data, folder_path)
    stimulus_types = np.sort(pinky_data['STIMULUS_TYPE'].unique().astype(int))
    if len(stimulus_types) > 1:
        create_pfit_multiple_stim(stimulus_types, pinky_data, folder_path)
    else:
        create_pfit_one_stim(pinky_data, folder_path)

    # Process log file
    logfiles = [file for file in listdir if file.startswith('AutoLOG') or file.startswith('LOG')]
    # Process the first logfile if the list is existed
    if logfiles:
        logfile = logfile_process(logfiles, folder_path)
    if not pinky_data.empty and not logfile.empty:
        MergeData = merge_data(pinky_data, logfile, folder_path)


if __name__ == '__main__':
    rats_folders = os.listdir(DATA_PATH)
    # chosen_rats = ['Lenon', 'Muki', 'Narkis']
    # chosen_rats = ['Muki']
    # chosen_rats = ['Lenon']
    chosen_rats = ['Narkis']
    for rat_folder in rats_folders:
        if rat_folder.split()[2] not in chosen_rats:
            continue
        rat_folder_path = os.path.join(DATA_PATH, rat_folder)
        date_list = os.listdir(rat_folder_path)[:-2]
        date_list = ['2023_12_05']
        for date in date_list:
            session_pipline(rat_folder_path, date)

        # pooled data
        start_date = '2023_11_21'
        end_date = '2023_12_05'
        create_pooled_results(rat_folder_path, start_date, end_date, [2, 10])
