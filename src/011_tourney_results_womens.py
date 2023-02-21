
import numpy as np
import pandas as pd
import sys

from utils import save2pkl, line_notify
from utils import WBASE_DIR

#==============================================================================
# preprocess tourney results womens
#==============================================================================

def main():

    # load csv
    TourneyCompactResults = pd.read_csv(f'{WBASE_DIR}/WNCAATourneyCompactResults.csv')
    SampleSubmission = pd.read_csv(f'{WBASE_DIR}/WSampleSubmissionStage2.csv')

    # split win & lose
    df_w = TourneyCompactResults.copy()
    df_l = TourneyCompactResults.copy()

    # change column names
    df_l.columns = ['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT']

    # merge
    df = df_w.append(df_l)
    del df_w, df_l

    # add target
    df['target'] = ((df['WScore']-df['LScore'])>0).astype(int)

    # add ID
    df['ID'] = df['Season'].astype(str) +'_'+ df['WTeamID'].astype(str) +'_'+ df['LTeamID'].astype(str)

    # split ID
    SampleSubmission['Season'] = SampleSubmission['ID'].apply(lambda x: x[:4]).astype(int)
    SampleSubmission['WTeamID'] = SampleSubmission['ID'].apply(lambda x: x[5:9]).astype(int)
    SampleSubmission['LTeamID'] = SampleSubmission['ID'].apply(lambda x: x[10:14]).astype(int)

    # add test flag
    df['is_test'] = False
    SampleSubmission['is_test'] = True

    # drop unnecessary columns
    df.drop(['DayNum', 'WLoc', 'NumOT','WScore','LScore'],axis=1,inplace=True)

    # merge 
    df = df.append(SampleSubmission.drop('Pred',axis=1))

    # save pkl
    save2pkl('../feats/tourney_result_womens.pkl', df)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()