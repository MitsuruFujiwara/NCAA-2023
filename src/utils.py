
import feather
import gc
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import pickle
import requests
import seaborn as sns

from sklearn.metrics import mean_squared_error
from time import sleep

#==============================================================================
# utils
#==============================================================================

NUM_FOLDS = 6

FEATS_EXCLUDED = ['WTeamID','LTeamID','target','ID','is_test','index']

COMPETITION_NAME_M = 'mens-march-mania-2022'
COMPETITION_NAME_W = 'womens-march-mania-2022'

# base directory
MBASE_DIR = '../input/mens/MDataFiles_Stage2'
WBASE_DIR = '../input/womens/WDataFiles_Stage2'

# dict for location
DICT_LOC = {'H':1, 'A':-1, 'N':0}

# dict for seed
DICT_SEED = {'a':0, 'b':1}

# dict for region
DICT_REGION = {'W':0, 'Y':1, 'X':2, 'Z':3}

# save pkl
def save2pkl(path, df):
    f = open(path, 'wb')
    pickle.dump(df, f)
    f.close

# load pkl
def loadpkl(path):
    f = open(path, 'rb')
    out = pickle.load(f)
    return out

# to feather
def to_feature(df, path):
    if df.columns.duplicated().sum()>0:
        raise Exception('duplicated!: {}'.format(df.columns[df.columns.duplicated()]))
    df.reset_index(inplace=True)
    df.columns = [c.replace('/', '-').replace(' ', '-') for c in df.columns]
    for c in df.columns:
        df[[c]].to_feather('{}_{}.feather'.format(path,c))
    return

# rmse
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# LINE Notify
def line_notify(message):
    f = open('../input/line_token.txt')
    token = f.read()
    f.close
    line_notify_token = token.replace('\n', '')
    line_notify_api = 'https://notify-api.line.me/api/notify'

    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    line_notify = requests.post(line_notify_api, data=payload, headers=headers)
    print(message)

# API submission https://github.com/KazukiOnodera/Home-Credit-Default-Risk/blob/master/py/utils.py
def submit(file_path, is_men=True, comment='from API'):
    if is_men:
        COMPETITION_NAME = COMPETITION_NAME_M
    else:
        COMPETITION_NAME = COMPETITION_NAME_W

    os.system('kaggle competitions submit -c {} -f {} -m "{}"'.format(COMPETITION_NAME,file_path,comment))
    sleep(360) # tekito~~~~
    tmp = os.popen('kaggle competitions submissions -c {} -v | head -n 2'.format(COMPETITION_NAME)).read()
    col, values = tmp.strip().split('\n')
    message = 'SCORE!!!\n'
    for i,j in zip(col.split(','), values.split(',')):
        message += '{}: {}\n'.format(i,j)
    line_notify(message.rstrip())

# save json
def to_json(data_dict, path):
    with open(path, 'w') as f:
        json.dump(data_dict, f, indent=4)

# plot scatter
def plot_scatter(y_true, y_pred, outputpath):
    plt.figure()
    plt.scatter(x=y_pred, y=y_true, s=10, alpha=0.3)
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    plt.savefig(outputpath)

# Display/plot feature importance
def save_imp(imp_df, path_png, path_csv):
    cols = imp_df[['feature', 'importance']].groupby('feature').mean().sort_values(by='importance', ascending=False)[:40].index
    best_features = imp_df.loc[imp_df.feature.isin(cols)]

    # for checking all importance
    imp_df_agg = imp_df.groupby('feature').sum()
    imp_df_agg.to_csv(path_csv)

    plt.figure(figsize=(8, 10))
    sns.barplot(x='importance', y='feature', data=best_features.sort_values(by='importance', ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(path_png)
