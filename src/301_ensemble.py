
import gc
import numpy as np
import pandas as pd
import sys

from scipy.optimize import minimize
from tqdm import tqdm

from utils import line_notify

#==============================================================================
# Ensemble
#==============================================================================

sub_path_lgbm = '../output/submission_lgbm.csv'
sub_path_xgb = '../output/submission_xgb.csv'
sub_path_cb = '../output/submission_cb.csv'

sub_path_avg = '../output/submission_avg.csv'

def main():

    # load csv
    sub_lgbm = pd.read_csv(sub_path_lgbm)
    sub_xgb  = pd.read_csv(sub_path_xgb)
    sub_cb   = pd.read_csv(sub_path_cb)

    # rename column
    sub_lgbm.columns = ['ID','Pred_lgbm']
    sub_xgb.columns  = ['ID','Pred_xgb']
    sub_cb.columns   = ['ID','Pred_cb']

    # merge submission
    sub = sub_lgbm.merge(sub_xgb,on='ID',how='left')
    sub = sub.merge(sub_cb,on='ID',how='left')

    # simple average
    sub['Pred'] = sub[['Pred_lgbm','Pred_xgb','Pred_cb']].mean(axis=1)

    # to probablity
    sub.loc[:,'Pred'] = (sub['Pred']-sub['Pred'].min())/(sub['Pred'].max()-sub['Pred'].min())

    # save csv
    sub[['ID','Pred']].to_csv(sub_path_avg, index=False)

    # LINE notify
    line_notify(f'{sys.argv[0]} done.')

if __name__ == '__main__':
    main()