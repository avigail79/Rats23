import os

WORK_PATH = os.getcwd()[:-5]
CODE_PATH = os.getcwd()
DATA_PATH = os.path.join(WORK_PATH, 'Data')

OPTIONS = {
            'sigmoidName': 'norm',
            'expType': 'equalAsymptote',
            'stimulusRange': [-90, 90]
        }