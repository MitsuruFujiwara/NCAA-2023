
import numpy as np
import pandas as pd
import sys

from utils import save2pkl, line_notify

#==============================================================================
# preprocess tourney results
#==============================================================================

def main():

    # load csv
    MTourneyCompactResults = pd.read_csv('../input/MNCAATourneyCompactResults.csv')
    WTourneyCompactResults = pd.read_csv('../input/WNCAATourneyCompactResults.csv')

    SampleSubmission = pd.read_csv('../input/SampleSubmission2023.csv')

    # split win & lose
    df_m_w = MTourneyCompactResults.copy()
    df_m_l = MTourneyCompactResults.copy()

    df_w_w = WTourneyCompactResults.copy()
    df_w_l = WTourneyCompactResults.copy()

    # change column names
    df_m_l.columns = ['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT']
    df_w_l.columns = ['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT']

    # merge
    df = pd.concat([df_m_w, df_m_l, df_w_w, df_w_l])

    del df_m_w, df_m_l, df_w_w, df_w_l

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
    df = pd.concat([df,SampleSubmission.drop('Pred',axis=1)])

    # add men/women flag
    df['is_women'] = (df['WTeamID']>3000).astype(int)

    # save pkl
    save2pkl('../feats/tourney_result.pkl', df)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()