
import numpy as np
import pandas as pd
import sys

from utils import loadpkl, to_feature, to_json, line_notify

#==============================================================================
# aggregate mens
#==============================================================================

def main():

    # load pkls
    df = loadpkl('../feats/tourney_result.pkl')
    df_season = loadpkl('../feats/season_result.pkl')
    Teams = loadpkl('../feats/teams.pkl')
    Seasons = loadpkl('../feats/seasons.pkl')
    TourneySeeds = loadpkl('../feats/seeds.pkl')
    MasseyOrdinals = loadpkl('../feats/massey_ordinals.pkl')
    TeamCoaches = loadpkl('../feats/coaches.pkl')
    Conferences = loadpkl('../feats/conferences.pkl')

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

    # merge teams
    Teams_w = Teams.copy()
    Teams_l = Teams.copy()

    Teams_w.columns = ['WTeamID', 'WTeams_FirstD1Season', 'WTeams_LastD1Season','WTeams_diff_D1Season']
    Teams_l.columns = ['LTeamID', 'LTeams_FirstD1Season', 'LTeams_LastD1Season','LTeams_diff_D1Season']

    df = df.merge(Teams_w,on='WTeamID',how='left')
    df = df.merge(Teams_l,on='LTeamID',how='left')

    # merge seasons
    df = df.merge(Seasons,on='Season',how='left')

    # merge tourney seeds
    TourneySeeds_w = TourneySeeds.copy() 
    TourneySeeds_l = TourneySeeds.copy()
    
    TourneySeeds_w.columns = ['Season', 'WTeamID', 'Wregion', 'Wseed_in_region', 'Wseed_in_region2']
    TourneySeeds_l.columns = ['Season', 'LTeamID', 'Lregion', 'Lseed_in_region', 'Lseed_in_region2']

    df = df.merge(TourneySeeds_w,on=['Season', 'WTeamID'],how='left')
    df = df.merge(TourneySeeds_l,on=['Season', 'LTeamID'],how='left')

    # merge massey ordinals
    MasseyOrdinals_w = MasseyOrdinals.copy()
    MasseyOrdinals_l = MasseyOrdinals.copy()

    MasseyOrdinals_w.columns = ['Season']+[f'W{c}' for c in MasseyOrdinals.columns if c not in ['Season']]
    MasseyOrdinals_l.columns = ['Season']+[f'L{c}' for c in MasseyOrdinals.columns if c not in ['Season']]

    df = df.merge(MasseyOrdinals_w,on=['Season','WTeamID'],how='left')
    df = df.merge(MasseyOrdinals_l,on=['Season','LTeamID'],how='left')

    # merge team coaches
    TeamCoaches_w = TeamCoaches.copy()
    TeamCoaches_l = TeamCoaches.copy()

    TeamCoaches_w.columns = ['Season', 'WTeamID', 'Wdays_coaches']
    TeamCoaches_l.columns = ['Season', 'LTeamID', 'Ldays_coaches']

    df = df.merge(TeamCoaches_w,on=['Season','WTeamID'],how='left')
    df = df.merge(TeamCoaches_l,on=['Season','LTeamID'],how='left')

    # merge conferences
    Conferences_w = Conferences.copy()
    Conferences_l = Conferences.copy()

    Conferences_w.columns = ['Season', 'WTeamID', 'WConfAbbrev']
    Conferences_l.columns = ['Season', 'LTeamID', 'LConfAbbrev']

    df = df.merge(Conferences_w,on=['Season','WTeamID'],how='left')
    df = df.merge(Conferences_l,on=['Season','LTeamID'],how='left')

    # add diff features
    df['diff_ConfAbbrev'] = df['WConfAbbrev']-df['LConfAbbrev']
    df['diff_days_coaches'] = df['Wdays_coaches']-df['Ldays_coaches']
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
    df['diff_OrdinalRank_mean'] = df['WOrdinalRank_mean']-df['LOrdinalRank_mean']
    df['diff_OrdinalRank_max'] = df['WOrdinalRank_max']-df['LOrdinalRank_max']
    df['diff_OrdinalRank_min'] = df['WOrdinalRank_min']-df['LOrdinalRank_min']
    df['diff_region'] = df['Wregion']-df['Lregion']
    df['diff_seed_in_region'] = df['Wseed_in_region']-df['Lseed_in_region']
    df['diff_seed_in_region2'] = df['Wseed_in_region2']-df['Lseed_in_region2']
    df['diff_Teams_FirstD1Season'] = df['WTeams_FirstD1Season']-df['LTeams_FirstD1Season']
    df['diff_Teams_LastD1Season'] = df['WTeams_LastD1Season']-df['LTeams_LastD1Season']
    df['diff_Teams_diff_D1Season'] = df['WTeams_diff_D1Season']-df['LTeams_diff_D1Season']

    # save as feather
    to_feature(df, '../feats/f101')

    # save feature name list
    features_json = {'features':df.columns.tolist()}
    to_json(features_json,'../configs/101_all_features.json')

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()