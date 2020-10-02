import requests
import os


DATAS_LOCAL_PATH = './DATAS/'
RAW_LOCAL_PATH = DATAS_LOCAL_PATH + 'RAW/'
TXT_LOCAL_PATH = RAW_LOCAL_PATH + 'valeursfoncieres-2019.txt'
CUR_LOCAL_PATH = DATAS_LOCAL_PATH + 'CUR/'
RESULT_LOCAL_PATH = './RESULTS/'
URL = 'https://www.data.gouv.fr/fr/datasets/r/3004168d-bec4-44d9-a781-ef16f41856a2'


def ensure_data_loaded():
    '''
    Ensure if data are already loaded. Download if missing
    '''
    if os.path.exists(TXT_LOCAL_PATH) == False:
        dl_data()
    else :
        print('Datas already douwnloaded.')


def dl_data ():
        print ('Downloading...')
        with open(TXT_LOCAL_PATH, "wb") as f:
            r = requests.get(URL)
            f.write(r.content)
        print ('Dataset dowloaded successfully.')


def check_folder ():
    PATH = [DATAS_LOCAL_PATH, RAW_LOCAL_PATH, RESULT_LOCAL_PATH, CUR_LOCAL_PATH]
    for p in PATH:
        if not os.path.exists(p):
            os.mkdir(p)
