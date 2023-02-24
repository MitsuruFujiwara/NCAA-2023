
import gc
import json
import numpy as np
import optuna
import pandas as pd
import sys
import warnings
import xgboost as xgb

from glob import glob
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from utils import line_notify, to_json
from utils import FEATS_EXCLUDED, NUM_FOLDS

#==============================================================================
# hyper parameter optimization by optuna
# https://github.com/pfnet/optuna/blob/master/examples/lightgbm_simple.py
#==============================================================================

warnings.filterwarnings('ignore')

# load datasets
CONFIGS = json.load(open('../configs/202_xgb.json'))

# load feathers
FILES = sorted(glob('../feats/f101_*.feather'))
DF = pd.concat([pd.read_feather(f) for f in tqdm(FILES)], axis=1)

# split train & test
TRAIN_DF = DF[~DF['is_test']]

del DF
gc.collect()

# use selected features
TRAIN_DF = TRAIN_DF[CONFIGS['features']]

FEATS = [f for f in TRAIN_DF.columns if f not in FEATS_EXCLUDED]

def objective(trial):

    # set data structure
    xgb_train = xgb.DMatrix(TRAIN_DF[FEATS],label=TRAIN_DF['target'])

    # ref: https://alphaimpact.jp/downloads/pydata20190927.pdf
    params = {'objective': 'binary:logistic',
              'learning_rate': 0.01,
#              'device': 'gpu',
              'max_depth': trial.suggest_int('max_depth', 3, 9),
              'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
              'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.01, 1.0),
              'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
              'lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 10.0),
              }

    # folds
    folds = GroupKFold(n_splits=NUM_FOLDS)

    # cv
    eval_dict = xgb.cv(params=params,
                       dtrain=xgb_train,
                       nfold=NUM_FOLDS,
                       folds=list(folds.split(TRAIN_DF[FEATS],groups=TRAIN_DF['Season'])),
                       num_boost_round=10000,
                       early_stopping_rounds=200,
                       verbose_eval=100,
                       seed=46,
                      )

    gc.collect()

    return min(eval_dict['test-logloss-mean'])

if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    # save result
    hist_df = study.trials_dataframe()
    hist_df.to_csv("../output/optuna_result_xgb.csv")

    # save json
    CONFIGS['params'] = trial.params
    to_json(CONFIGS, '../configs/202_xgb.json')

    line_notify('{} finished. Value: {}'.format(sys.argv[0],trial.value))
