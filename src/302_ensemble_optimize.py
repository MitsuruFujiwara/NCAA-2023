
import gc
import numpy as np
import pandas as pd
import sys

from scipy.optimize import minimize
from sklearn.metrics import log_loss
from tqdm import tqdm

from utils import line_notify

#==============================================================================
# Ensemble optimize for out of fold
#==============================================================================

sub_path_lgbm = '../output/submission_lgbm.csv'
sub_path_xgb = '../output/submission_xgb.csv'
sub_path_cb = '../output/submission_cb.csv'

oof_path_lgbm = '../output/oof_lgbm.csv'
oof_path_xgb = '../output/oof_xgb.csv'
oof_path_cb = '../output/oof_cb.csv'

sub_path_avg = '../output/submission_opt.csv'

def main():

    # load csv
    sub_lgbm = pd.read_csv(sub_path_lgbm)
    sub_xgb  = pd.read_csv(sub_path_xgb)
    sub_cb   = pd.read_csv(sub_path_cb)

    oof_lgbm = pd.read_csv(oof_path_lgbm)
    oof_xgb  = pd.read_csv(oof_path_xgb)
    oof_cb   = pd.read_csv(oof_path_cb)

    # rename column
    sub_lgbm.columns = ['ID','Pred_lgbm']
    sub_xgb.columns  = ['ID','Pred_xgb']
    sub_cb.columns   = ['ID','Pred_cb']

    oof_lgbm.columns = ['ID','target','Pred_lgbm']
    oof_xgb.columns  = ['ID','target','Pred_xgb']
    oof_cb.columns   = ['ID','target','Pred_cb']

    # merge submission
    sub = sub_lgbm.merge(sub_xgb,on='ID',how='left')
    sub = sub.merge(sub_cb,on='ID',how='left')

    oof = oof_lgbm.merge(oof_xgb,on=['ID','target'],how='left')
    oof = oof.merge(oof_cb,on=['ID','target'],how='left')

    # predictions to use
    cols_pred = ['Pred_lgbm','Pred_xgb','Pred_cb']

    # objective function for scipy optimize
    def obj_func(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        final_prediction = 0
        for weight, c in zip(weights, cols_pred):
                final_prediction += weight*oof[c]

        return log_loss(oof['target'], final_prediction)    

    # Optimization runs 100 times.
    lls = []
    wghts = []
    print('Optimization runs 100 times...')
    for i in tqdm(range(100)):
        starting_values = np.random.uniform(size=len(cols_pred))
        # cons are given as constraints.
        cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
        bounds = [(0,1)]*len(cols_pred)
        
        res = minimize(obj_func, 
                       starting_values, 
                       constraints=cons,
                       bounds = bounds, 
                       method='SLSQP')

        lls.append(res['fun'])
        wghts.append(res['x'])

    # get weights
    bestSC = np.min(lls)
    w = wghts[np.argmin(lls)] # [8.26472566e-01 1.41295664e-01 5.84757810e-04 3.16470122e-02]
    print('\n Ensemble Score: {best_score:.7f}'.format(best_score=bestSC))
    print('weights: {}'.format(w))

    # calc weighted average
    sub['Pred'] = w[0]*sub['Pred_lgbm']+w[1]*sub['Pred_xgb']+w[2]*sub['Pred_cb']
    oof['Pred'] = w[0]*oof['Pred_lgbm']+w[1]*oof['Pred_xgb']+w[2]*oof['Pred_cb']

    # to probablity
    sub.loc[:,'Pred'] = (sub['Pred']-sub['Pred'].min())/(sub['Pred'].max()-sub['Pred'].min())
    oof.loc[:,'Pred'] = (oof['Pred']-oof['Pred'].min())/(oof['Pred'].max()-oof['Pred'].min())

    # clip prediction
    sub['Pred'].clip(0.05,0.95,inplace=True)
    oof['Pred'].clip(0.05,0.95,inplace=True)

    # calc score
    full_score = round(log_loss(oof['target'], oof['Pred']),6)

    # save csv
    sub[['ID','Pred']].to_csv(sub_path_avg, index=False)

    # LINE notify
    line_notify(f'{sys.argv[0]} done. Full logloss: {full_score}')

if __name__ == '__main__':
    main()