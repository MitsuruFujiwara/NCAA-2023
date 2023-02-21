
import numpy as np
import pandas as pd
import sys

from utils import loadpkl, to_feature, to_json, line_notify

#==============================================================================
# aggregate womens
#==============================================================================

def main():

    # load pkls
    df = loadpkl('../feats/tourney_result_womens.pkl')
    df_season = loadpkl('../feats/season_result_womens.pkl')
    Seasons = loadpkl('../feats/seasons_womens.pkl')
    TourneySeeds = loadpkl('../feats/seeds_womens.pkl')
    Conferences = loadpkl('../feats/conferences_womens.pkl')

    # drop results before 2010
    df = df[df['Season']>=2010]

    # merge season result
    df_season_w = df_season.copy()
    df_season_l = df_season.copy()

    df_season_w.columns = ['Season']+[f'W{c}' for c in df_season.columns if c not in ['Season']]
    df_season_l.columns = ['Season']+[f'L{c}' for c in df_season.columns if c not in ['Season']]

    df = df.merge(df_season_w,on=['Season','WTeamID'],how='left')
    df = df.merge(df_season_l,on=['Season','LTeamID'],how='left')

    # merge shifted season result
    df_season_shift_w = df_season.copy()
    df_season_shift_l = df_season.copy()

    df_season_shift_w['Season'] += 1
    df_season_shift_l['Season'] += 1

    df_season_shift_w.columns = ['Season','WTeamID']+[f'W_Shift_{c}' for c in df_season.columns if c not in ['Season','TeamID']]
    df_season_shift_l.columns = ['Season','LTeamID']+[f'L_Shift_{c}' for c in df_season.columns if c not in ['Season','TeamID']]

    df = df.merge(df_season_shift_w,on=['Season','WTeamID'],how='left')
    df = df.merge(df_season_shift_l,on=['Season','LTeamID'],how='left')

    # merge seasons
    df = df.merge(Seasons,on='Season',how='left')

    # merge tourney seeds
    TourneySeeds_w = TourneySeeds.copy() 
    TourneySeeds_l = TourneySeeds.copy()
    
    TourneySeeds_w.columns = ['Season', 'WTeamID', 'Wregion', 'Wseed_in_region', 'Wseed_in_region2']
    TourneySeeds_l.columns = ['Season', 'LTeamID', 'Lregion', 'Lseed_in_region', 'Lseed_in_region2']

    df = df.merge(TourneySeeds_w,on=['Season', 'WTeamID'],how='left')
    df = df.merge(TourneySeeds_l,on=['Season', 'LTeamID'],how='left')

    # merge conferences
    Conferences_w = Conferences.copy()
    Conferences_l = Conferences.copy()

    Conferences_w.columns = ['Season', 'WTeamID', 'WConfAbbrev']
    Conferences_l.columns = ['Season', 'LTeamID', 'LConfAbbrev']

    df = df.merge(Conferences_w,on=['Season','WTeamID'],how='left')
    df = df.merge(Conferences_l,on=['Season','LTeamID'],how='left')

    # add diff features
    df['diff_ConfAbbrev'] = df['WConfAbbrev']-df['LConfAbbrev']
    df['diff_Score_mean'] = df['WScore_mean']-df['LScore_mean']
    df['diff_Score_sum'] = df['WScore_sum']-df['LScore_sum']
    df['diff_Score_max'] = df['WScore_max']-df['LScore_max']
    df['diff_Score_min'] = df['WScore_min']-df['LScore_min']
    df['diff_ScoreGap_sum'] = df['WScoreGap_sum']-df['LScoreGap_sum']
    df['diff_ScoreGap_mean'] = df['WScoreGap_mean']-df['LScoreGap_mean']
    df['diff_ScoreGap_max'] = df['WScoreGap_max']-df['LScoreGap_max']
    df['diff_ScoreGap_min'] = df['WScoreGap_min']-df['LScoreGap_min']
    df['diff_diff_win_lose'] = df['Wdiff_win_lose']-df['Ldiff_win_lose']
    df['diff_Stl_sum'] = df['WStl_sum']-df['LStl_sum']
    df['diff_DR_mean'] = df['WDR_mean']-df['LDR_mean']
    df['diff_TO_mean'] = df['WTO_mean']-df['LTO_mean']
    df['diff_OR_std'] = df['WOR_std']-df['LOR_std']
    df['diff_FGM3_std'] = df['WFGM3_std']-df['LFGM3_std']
    df['diff_PF_mean'] = df['WPF_mean']-df['LPF_mean']
    df['diff_DayNum_mean'] = df['WDayNum_mean']-df['LDayNum_mean']
    df['diff_region'] = df['Wregion']-df['Lregion']
    df['diff_seed_in_region'] = df['Wseed_in_region']-df['Lseed_in_region']
    df['diff_seed_in_region2'] = df['Wseed_in_region2']-df['Lseed_in_region2']
    
    # save as feather
    to_feature(df, '../feats/f102')

    # save feature name list
    features_json = {'features':df.columns.tolist()}
    to_json(features_json,'../configs/102_all_features_womens.json')

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()