import os
def pipline():
    WORK_PATH = os.getcwd()[:-5]
    CODE_PATH = os.getcwd()
    DATA_PATH = os.path.join(WORK_PATH, 'Data')

    return WORK_PATH


if __name__ == '__main__':
    print(pipline())