import re
from scipy.io import savemat
import pandas as pd
import os
import scipy.io
import glob
import csv
from copy import deepcopy
from datetime import datetime


def is_date_within_range(date, start, end):
    date = datetime.strptime(date.replace('_', '-'), '%Y-%m-%d')
    start = datetime.strptime(start.replace('_', '-'), '%Y-%m-%d')
    end = datetime.strptime(end.replace('_', '-'), '%Y-%m-%d')
    return start <= date <= end


#### Pinky Protocols ####
def convert_txt_pinky_protocol_to_df(pinkyfile_path):
    columns = ['Trial', 'Rat Decison', 'STIMULUS_TYPE', 'HEADING_DIRECTION', 'TrialBeginRealTime', 'AudioStartRealTime',
               'HeadEnterCenterRealTime', 'StimulusStartRealTime', 'RobotEndMovingForwardRealTime',
               'CenterRewardRealTime',
               'GoCueSoundRealTime', 'RatDecisionRealTime', 'SideRewardSoundRealTime', 'RightRewardRealTime',
               'RobotStartMovingBackwardRealTime', 'TotalManualReward']
    with open(pinkyfile_path, 'r') as f:
        lines = f.readlines()

    current_parameter_dict = {}
    parameter_index = 0
    trial_num = 0
    first_trial = True
    df = pd.DataFrame()
    trial_dict = {}

    for line in lines:
        split_line = line.split(":")
        field_name = split_line[0].strip()
        if field_name not in columns:
            continue

        field_val = split_line[1].strip()
        try:
            field_val = float(field_val)
        except ValueError:
            pass

        if field_name == "Trial" and not first_trial:
            # df = df.append(trial_dict, ignore_index=True)
            df = pd.concat([df, pd.DataFrame(trial_dict, index=[0])], ignore_index=True)
            parameter_index = 0
            trial_num += 1
            trial_dict = {}
        else:
            first_trial = False

        trial_dict[field_name] = field_val
        parameter_index += 1
    df = pd.concat([df, pd.DataFrame(trial_dict, index=[0])], ignore_index=True)

    return df


def pinky_process(pinky_files, folder_path):
    columns = ['Trial', 'Rat Decison', 'STIMULUS_TYPE', 'HEADING_DIRECTION', 'TrialBeginRealTime', 'AudioStartRealTime',
               'HeadEnterCenterRealTime', 'StimulusStartRealTime', 'RobotEndMovingForwardRealTime',
               'CenterRewardRealTime',
               'GoCueSoundRealTime', 'RatDecisionRealTime', 'SideRewardSoundRealTime', 'RightRewardRealTime',
               'RobotStartMovingBackwardRealTime', 'TotalManualReward']
    pinky_data = pd.DataFrame(columns=columns)
    for f in pinky_files:
        pinky_file_path = os.path.join(folder_path, f)
        curr_data = convert_txt_pinky_protocol_to_df(pinky_file_path)
        pinky_data = pd.concat([pinky_data, curr_data])

    # # save the data to a .mat file
    # mat_file_path = os.path.join(folder_path, 'pinky_data.mat')
    # if os.path.isfile(mat_file_path):
    #     user_input = input(f"A file already exists at {mat_file_path}. Do you want to overwrite it? (yes/no): ")
    #     if user_input.lower() == "no":
    #         return df
    # savemat(mat_file_path, df.to_dict(orient='list'))

    # save the data to a .csv file
    date = folder_path.split('\\')[-1]
    rat_name = folder_path.split('\\')[-2][5:]
    csv_file_path = os.path.join(folder_path, f'PinkyData {date} {rat_name}.csv')
    # todo: after finish
    # if os.path.isfile(csv_file_path):
        # user_input = input(f"A file already exists at {csv_file_path}. Do you want to overwrite it? press space to continue")
        # if user_input.lower() == " ":
        #     return pinky_data
    pinky_data.to_csv(csv_file_path, index=False)
    return pinky_data


#### NL ####
def get_MSB_from_decimal(decimal_num):
    """Get the MSB value from a decimal number."""
    binary_str = bin(decimal_num)[2:]
    return int(binary_str[0]) * (2 ** (len(binary_str) - 1))


def trial_num_health_check(trial_bin, counter15_bin, counter15):
    # positions = [pos for pos, char in enumerate(trial_bin, start=0) if char == '1']
    for pos, char in enumerate(trial_bin, start=0):  # health check
        if char == '1':  # if there is 1 in the trial_bin but not in the counter15_bin, it's problem!
            if counter15_bin[pos] != '1':
                print(f'Trial no.{counter15}: counter15_bin: {counter15_bin}, trial_bin: {trial_bin}')


def add_row_to_logfile(logfile, row):
    logfile.loc[len(logfile)] = row


def process_log_row(row):
    tokens = row[2].split()
    if len(tokens) > 4 and tokens[4] == 'Digital':
        trial_time = row[1]
        state = tokens[7][6:]
        channel = int(tokens[6][3:])
        return channel, state, trial_time
    return None, None, None


def convert_logfile_to_df(logfile_path):
    # Constants for clarity
    TRIAL_BIN_INITIAL = '0000000000000'
    BINARY_FORMAT = '013b'
    MSB_LIST = [get_MSB_from_decimal(i) for i in range(1, 500)]  # Pre-calculate msb_list if it remains constant

    logfile_raw = pd.read_csv(logfile_path, header=None, on_bad_lines='skip')
    counter15 = 0
    trial_bin = TRIAL_BIN_INITIAL
    logfile = pd.DataFrame(columns=[
        'Trial number',
        'Trial number - binary',
        'Expected MSB',
        'Trial number excepted - bins',
        'Trial number - bins ',
        'Start time',
        'End time'
    ])
    start_rec = 0
    for row in logfile_raw.itertuples(index=False):
        channel, state, trial_time = process_log_row(row)
        if channel is None:
            if row[2].endswith("to the PC clock"):
                start_rec = 1
                continue
            continue

        if state == '1':
            if channel < 15:
                trial_bin = trial_bin[:channel - 1] + state + trial_bin[channel:]
                start_time = trial_time
            elif channel == 15:
                if not start_rec:
                    continue
                end_time = trial_time
                counter15 += 1
                counter15_bin = format(counter15, BINARY_FORMAT)
                trial_bin = trial_bin[::-1]
                trial_bin_num = int(trial_bin, 2)
                expected_bin = MSB_LIST[counter15 - 1]
                trial_num_health_check(trial_bin, counter15_bin, counter15)
                add_row_to_logfile(logfile,
                                   [counter15, trial_bin_num, expected_bin, counter15_bin, trial_bin, start_time,
                                    end_time])
                trial_bin = TRIAL_BIN_INITIAL

    return logfile


def logfile_process(logfiles, folder_path):
    logfile = pd.DataFrame(columns=['Trial number', 'Trial number - binary', 'Expected MSB',
                                    'Trial number excepted - bins', 'Trial number - bins ', 'Start time', 'End time'])
    for f in logfiles:
        logfile_path = os.path.join(folder_path, f)
        curr_data = convert_logfile_to_df(logfile_path)
        logfile = pd.concat([logfile, curr_data])

    # Adjust start and end times
    first_start_time = logfile['Start time'].iloc[0]
    logfile['Start time'] -= first_start_time
    logfile['End time'] -= first_start_time

    # Save the processed logfile
    date = folder_path.split('\\')[-1]
    rat_name = folder_path.split('\\')[-2][5:]
    logfile.to_csv(os.path.join(folder_path, f'NLDAta {date} {rat_name}.csv'), index=False)
    return logfile


####  ####
def merge_data(pinky_data, logfile, folder_path):
    MergeData = deepcopy(pinky_data)
    MergeData.insert(4, 'StartTimeOnSet', logfile['Start time'])
    MergeData.insert(17, 'EndTimeOnSet', logfile['End time'])
    MergeData.to_csv(os.path.join(folder_path, 'MergeData.csv'), index=False)
    return MergeData


def convert_txt_protocol_experiment_to_df(file_path):
    columns = ['Trial', 'Rat Decison', 'STIMULUS_TYPE', 'HEADING_DIRECTION', 'TrialBeginRealTime', 'AudioStartRealTime',
               'HeadEnterCenterRealTime', 'StimulusStartRealTime', 'RobotEndMovingForwardRealTime',
               'CenterRewardRealTime',
               'GoCueSoundRealTime', 'RatDecisionRealTime', 'SideRewardSoundRealTime', 'RightRewardRealTime',
               'RobotStartMovingBackwardRealTime', 'TotalManualReward']
    with open(file_path, 'r') as f:
        lines = f.readlines()

    current_parameter_dict = {}
    parameter_index = 0
    trial_num = 0
    first_trial = True
    df = pd.DataFrame()
    trial_dict = {}

    for line in lines:
        split_line = line.split(":")
        field_name = split_line[0].strip()
        if field_name not in columns:
            continue

        field_val = split_line[1].strip()
        try:
            field_val = float(field_val)
        except ValueError:
            pass

        if field_name == "Trial" and not first_trial:
            # df = df.append(trial_dict, ignore_index=True)
            df = pd.concat([df, pd.DataFrame(trial_dict, index=[0])], ignore_index=True)
            parameter_index = 0
            trial_num += 1
            trial_dict = {}
        else:
            first_trial = False

        trial_dict[field_name] = field_val
        parameter_index += 1
    # df = df.append(trial_dict, ignore_index=True) # add also the final trial to the saved data
    df = pd.concat([df, pd.DataFrame(trial_dict, index=[0])], ignore_index=True)

    # save the data to a .mat file
    mat_file_path = file_path[:-3] + 'mat'
    if os.path.isfile(mat_file_path):
        user_input = input(f"A file already exists at {mat_file_path}. Do you want to overwrite it? (yes/no): ")
        if user_input.lower() != "yes":
            return
    savemat(mat_file_path, df.to_dict(orient='list'))

    # save the data to a .csv file
    csv_file_path = file_path[:-3] + 'csv'
    if os.path.isfile(csv_file_path):
        user_input = input(f"A file already exists at {csv_file_path}. Do you want to overwrite it? (yes/no): ")
        if user_input.lower() != "yes":
            return
    df.to_csv(csv_file_path, index=False)


def add_general_time_from_log_to_pinky(log, pinky):
    all_data = pinky
    list_trail_start = log[log['Channel'] != 15]['Trial_time'].tolist()
    list_move_backward = log[log['Channel'] == 15]['Trial_time'].tolist()

    all_data.insert(loc=4, column='TIME_START', vlaue=list_trail_start)
    all_data.insert(loc=15, column='TIME_BACKWARD', value=list_move_backward)
    return all_data


# if __name__ == '__main__':
    # convert_txt_pinky_protocol_to_df(r'C:\Zaidel\Rats\NL\Data\31 - T\2022_07_18\2022_07_18_13-12 Rat 31 - T.txt')
    # logfile_procsses('LOG_Animal001_18_07_2022 T - 31.txt', r'C:\Zaidel\Rats\NL\Data\31 - T\2022_07_18')