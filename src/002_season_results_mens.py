
import numpy as np
import pandas as pd
import sys

from utils import save2pkl, line_notify
from utils import MBASE_DIR, DICT_LOC

#==============================================================================
# preprocess season results mens
#==============================================================================

def main():

    # load csv
    RegularSeasonDetailedResults = pd.read_csv(f'{MBASE_DIR}/MRegularSeasonDetailedResults.csv')

    # add score gap
    RegularSeasonDetailedResults['WScoreGap'] = RegularSeasonDetailedResults['WScore']-RegularSeasonDetailedResults['LScore']
    RegularSeasonDetailedResults['LScoreGap'] = RegularSeasonDetailedResults['LScore']-RegularSeasonDetailedResults['WScore']

    # label encoding WLoc & add LLoc
    RegularSeasonDetailedResults['WLoc'] = RegularSeasonDetailedResults['WLoc'].map(DICT_LOC)
    RegularSeasonDetailedResults['LLoc'] = -RegularSeasonDetailedResults['WLoc']

    # number of win & lose
    df_num_w = RegularSeasonDetailedResults.groupby(['Season','WTeamID']).count()['DayNum'].reset_index()
    df_num_l = RegularSeasonDetailedResults.groupby(['Season','LTeamID']).count()['DayNum'].reset_index()
    df_num_w.rename(columns={'WTeamID':'TeamID','DayNum':'NumWin'},inplace=True)
    df_num_l.rename(columns={'LTeamID':'TeamID','DayNum':'NumLose'},inplace=True)

    # merge win & lose
    df = df_num_w.merge(df_num_l,on=['Season','TeamID'],how='outer')

    # game stats
    df_w = RegularSeasonDetailedResults.copy()
    df_l = RegularSeasonDetailedResults.copy()

    df_l.columns = ['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'LLoc',
                    'NumOT', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR',
                    'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3',
                    'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF',
                    'LScoreGap', 'WScoreGap', 'WLoc']

    # merge
    tmp_df = df_w.append(df_l)

    # drop unnecessary columns
    tmp_df = tmp_df[['Season', 'DayNum','WTeamID', 'WScore', 'WScoreGap', 'WLoc', 'NumOT',
                     'WFGM', 'WFGA', 'WFGM3', 'WFGA3','WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 
                     'WTO', 'WStl', 'WBlk', 'WPF']]

    # rename columns
    tmp_df.columns = ['Season', 'DayNum','TeamID', 'Score', 'ScoreGap', 'Loc', 'NumOT',
                      'FGM', 'FGA', 'FGM3', 'FGA3','FTM', 'FTA', 'OR', 'DR', 'Ast', 
                      'TO', 'Stl', 'Blk', 'PF']

    # aggregate
    df_mean = tmp_df.groupby(['Season','TeamID']).mean()
    df_std = tmp_df.groupby(['Season','TeamID']).std()
    df_max = tmp_df.groupby(['Season','TeamID']).max()
    df_min = tmp_df.groupby(['Season','TeamID']).min()
    df_sum = tmp_df.groupby(['Season','TeamID']).sum()

    df_mean.columns = [f'{c}_mean' for c in df_mean.columns]
    df_std.columns = [f'{c}_std' for c in df_std.columns]
    df_max.columns = [f'{c}_max' for c in df_max.columns]
    df_min.columns = [f'{c}_min' for c in df_min.columns]
    df_sum.columns = [f'{c}_sum' for c in df_sum.columns]

    df_mean.reset_index(inplace=True)
    df_std.reset_index(inplace=True)
    df_max.reset_index(inplace=True)
    df_min.reset_index(inplace=True)
    df_sum.reset_index(inplace=True)

    df_stat = df_mean.merge(df_std,on=['Season','TeamID'],how='outer')
    df_stat = df_stat.merge(df_max,on=['Season','TeamID'],how='outer')
    df_stat = df_stat.merge(df_min,on=['Season','TeamID'],how='outer')
    df_stat = df_stat.merge(df_sum,on=['Season','TeamID'],how='outer')

    # merge
    df = df.merge(df_stat,on=['Season','TeamID'],how='outer')

    # add features ref: https://github.com/kazukim10/kaggle_MarchMania2021
    df['diff_win_lose'] = df['NumWin'] - df['NumLose']
    df['WinRatio'] = df['NumWin'] / (df['NumWin']+df['NumLose'])
    
    # FGM/FGA
    df['FGRatio'] = df['FGM_mean'] / df['FGA_mean']

    # FGM3/FGA3
    df['FG3Ratio'] = df['FGM3_mean'] / df['FGA3_mean']

    # FTM/FTA
    df['FTRatio'] = df['FTM_mean'] / df['FTA_mean']

    # EV(expect value)
    df['EV'] = df['FGM_mean']*df['FGRatio']+df['FGM3_mean']*df['FG3Ratio']+df['FTM_mean']*df['FTRatio']

    # TR
    df['TR'] = df['OR_mean']+df['DR_mean']

    # TS%
    df['TS%'] = df['Score_mean']/(2*(df['FGA_mean'] +0.44*df['FTA_mean']))

    # Four-Factor
    df['eFG'] = (df['FGM_mean'] +0.5*df['FGM3_mean'])/df['FGA_mean']
    df['OR%'] = df['OR_mean']/df['FGA_mean']
    df['TO%'] = df['TO_mean']/(df['FGA_mean'] +0.44*df['FTA_mean'] +df['TO_mean'])
    df['FTR'] = df['FTA_mean']/df['FGA_mean']

    # PPP/POSS
    df['POSS'] = df['FGA_mean'] +0.44*df['FTA_mean'] +df['TO_mean']
    df['PPP'] = df['Score_mean']/df['POSS']

    # save pkl
    save2pkl('../feats/season_result_mens.pkl', df)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()