
import catboost as cb
import gc
import json
import numpy as np
import pandas as pd
import sys
import warnings

from glob import glob
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from tqdm import tqdm

from utils import line_notify, save_imp
from utils import NUM_FOLDS, FEATS_EXCLUDED

#==============================================================================
# Train CatBoost
#==============================================================================

warnings.filterwarnings('ignore')

configs = json.load(open('../configs/203_cb.json'))

feats_path = '../feats/f101_*.feather'

sub_path = '../output/submission_cb.csv'
oof_path = '../output/oof_cb.csv'

model_path = '../models/cb_'

imp_path_png = '../imp/cb_importances.png'
imp_path_csv = '../imp/feature_importance_cb.csv'

params = configs['params']

params['learning_rate'] = 0.01
params['od_type'] = 'Iter'
params['loss_function'] = 'Logloss'
params['train_dir'] = '../output/catboost_info'
params['early_stopping_rounds'] = 200
params['random_seed'] = 47
params['verbose_eval'] = 100

def main():
    # load feathers
    files = sorted(glob(feats_path))
    df = pd.concat([pd.read_feather(f) for f in tqdm(files)], axis=1)

    # drop unused features
    df = df[configs['features']]

    # split train & test
    train_df = df[~df['is_test']]
    test_df = df[df['is_test']]

    del df
    gc.collect()

    # Cross validation
    folds = GroupKFold(n_splits=NUM_FOLDS)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])

    # for importance
    imp_df = pd.DataFrame()

    # features to use
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], groups=train_df['Season'])):
        
        # split train & test
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        print(valid_x['Season'].unique())

        # set data structure
        cb_train = cb.Pool(train_x,label=train_y)

        cb_test = cb.Pool(valid_x,label=valid_y)

        # train
        reg = cb.train(
                       cb_train,
                       params,
                       num_boost_round=10000,
                       eval_set=cb_test
                       )

        # save model
        reg.save_model(f'{model_path}{n_fold}.txt')

        # save predictions
        oof_preds[valid_idx] = reg.predict(valid_x,prediction_type='Probability')[:,1]
        sub_preds += reg.predict(test_df[feats],prediction_type='Probability')[:,1] / folds.n_splits

        # save importances
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = feats
        fold_importance_df['importance'] = np.log1p(reg.feature_importances_)
        imp_df = pd.concat([imp_df, fold_importance_df], axis=0)

        # calc fold score
        fold_score = log_loss(valid_y, oof_preds[valid_idx])

        print(f'Fold {n_fold+1} logloss: {fold_score}')

        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()

    # Full score and LINE Notify
    full_score = round(log_loss(train_df['target'], oof_preds),6)
    line_notify(f'Full logloss: {full_score}')

    # save importance
    save_imp(imp_df,imp_path_png,imp_path_csv)

    # save prediction
    train_df.loc[:,'Pred'] = oof_preds
    test_df.loc[:,'Pred'] = sub_preds

    # to probablity
    test_df.loc[:,'Pred'] = (test_df['Pred']-test_df['Pred'].min())/(test_df['Pred'].max()-test_df['Pred'].min())

    # save csv
    train_df[['ID','target','Pred']].to_csv(oof_path, index=False)
    test_df[['ID','Pred']].to_csv(sub_path, index=False)

    # LINE notify
    line_notify(f'{sys.argv[0]} done.')

if __name__ == '__main__':
    main()
