
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

    # add gap features
    cols_gap_season = ['DayNum','Score','ScoreGap','Loc','NumOT','FGM',
                       'FGA','FGM3','FGA3','FTM','FTA','OR','DR',
                       'Ast', 'TO', 'Stl', 'Blk', 'PF']

    for c in cols_gap_season:
        df[f'diff_{c}_mean'] = df[f'W{c}_mean']-df[f'L{c}_mean']
        df[f'diff_{c}_std'] = df[f'W{c}_std']-df[f'L{c}_std']
        df[f'diff_{c}_max'] = df[f'W{c}_max']-df[f'L{c}_max']
        df[f'diff_{c}_min'] = df[f'W{c}_min']-df[f'L{c}_min']
        df[f'diff_{c}_sum'] = df[f'W{c}_sum']-df[f'L{c}_sum']

    # add gap features ratio
    cols_gap_season_ratio = ['diff_win_lose','WinRatio','FGRatio','FG3Ratio','FTRatio','EV','TR','TS%',
                             'eFG','OR%','TO%','FTR','POSS','PPP','PTS']
    
    for c in cols_gap_season_ratio:
        df[f'diff_{c}'] = df[f'W{c}']-df[f'L{c}']

    # merge shifted season result
    df_season_shift_w = df_season.copy()
    df_season_shift_l = df_season.copy()

    df_season_shift_w['Season'] += 1
    df_season_shift_l['Season'] += 1

    df_season_shift_w.columns = ['Season','WTeamID']+[f'W_Shift_{c}' for c in df_season.columns if c not in ['Season','TeamID']]
    df_season_shift_l.columns = ['Season','LTeamID']+[f'L_Shift_{c}' for c in df_season.columns if c not in ['Season','TeamID']]

    df = df.merge(df_season_shift_w,on=['Season','WTeamID'],how='left')
    df = df.merge(df_season_shift_l,on=['Season','LTeamID'],how='left')

    # add gap features
    cols_gap_season = ['DayNum','Score','ScoreGap','Loc','NumOT','FGM',
                       'FGA','FGM3','FGA3','FTM','FTA','OR','DR',
                       'Ast', 'TO', 'Stl', 'Blk', 'PF']

    for c in cols_gap_season:
        df[f'diff_Shift_{c}_mean'] = df[f'W_Shift_{c}_mean']-df[f'L_Shift_{c}_mean']
        df[f'diff_Shift_{c}_std'] = df[f'W_Shift_{c}_std']-df[f'L_Shift_{c}_std']
        df[f'diff_Shift_{c}_max'] = df[f'W_Shift_{c}_max']-df[f'L_Shift_{c}_max']
        df[f'diff_Shift_{c}_min'] = df[f'W_Shift_{c}_min']-df[f'L_Shift_{c}_min']
        df[f'diff_Shift_{c}_sum'] = df[f'W_Shift_{c}_sum']-df[f'L_Shift_{c}_sum']

    # add gap features ratio
    cols_gap_season_ratio = ['diff_win_lose','WinRatio','FGRatio','FG3Ratio','FTRatio','EV','TR','TS%',
                             'eFG','OR%','TO%','FTR','POSS','PPP','PTS']
    
    for c in cols_gap_season_ratio:
        df[f'diff_Shift_{c}'] = df[f'W_Shift_{c}']-df[f'L_Shift_{c}']

    # merge teams
    Teams_w = Teams.copy()
    Teams_l = Teams.copy()

    Teams_w.columns = ['WTeamID', 'WTeams_FirstD1Season', 'WTeams_LastD1Season','WTeams_diff_D1Season']
    Teams_l.columns = ['LTeamID', 'LTeams_FirstD1Season', 'LTeams_LastD1Season','LTeams_diff_D1Season']

    df = df.merge(Teams_w,on='WTeamID',how='left')
    df = df.merge(Teams_l,on='LTeamID',how='left')

    # merge seasons
    df = df.merge(Seasons,on=['Season','is_women'],how='left')

    # add gap feature
    cols_gap_teams = ['Teams_FirstD1Season', 'Teams_LastD1Season','Teams_diff_D1Season']

    for c in cols_gap_teams:
        df[f'diff_{c}'] = df[f'W{c}']-df[f'L{c}']

    # merge tourney seeds
    TourneySeeds_w = TourneySeeds.copy() 
    TourneySeeds_l = TourneySeeds.copy()
    
    TourneySeeds_w.columns = ['Season', 'WTeamID', 'Wregion', 'Wseed_in_region', 'Wseed_in_region2']
    TourneySeeds_l.columns = ['Season', 'LTeamID', 'Lregion', 'Lseed_in_region', 'Lseed_in_region2']

    df = df.merge(TourneySeeds_w,on=['Season', 'WTeamID'],how='left')
    df = df.merge(TourneySeeds_l,on=['Season', 'LTeamID'],how='left')

    # add gap features
    cols_gap_seeds = ['region', 'seed_in_region', 'seed_in_region2']

    for c in cols_gap_seeds:
        df[f'diff_{c}'] = df[f'W{c}']-df[f'L{c}']

    # merge massey ordinals
    MasseyOrdinals_w = MasseyOrdinals.copy()
    MasseyOrdinals_l = MasseyOrdinals.copy()

    MasseyOrdinals_w.columns = ['Season']+[f'W{c}' for c in MasseyOrdinals.columns if c not in ['Season']]
    MasseyOrdinals_l.columns = ['Season']+[f'L{c}' for c in MasseyOrdinals.columns if c not in ['Season']]

    df = df.merge(MasseyOrdinals_w,on=['Season','WTeamID'],how='left')
    df = df.merge(MasseyOrdinals_l,on=['Season','LTeamID'],how='left')

    # add gap features
    cols_ordinal_rank = [c for c in MasseyOrdinals.columns if c not in ['Season','TeamID']]
    for c in cols_ordinal_rank:
        df[f'diff_{c}'] = df[f'W{c}']-df[f'L{c}']

    # merge team coaches
    TeamCoaches_w = TeamCoaches.copy()
    TeamCoaches_l = TeamCoaches.copy()

    TeamCoaches_w.columns = ['Season', 'WTeamID', 'Wdays_coaches']
    TeamCoaches_l.columns = ['Season', 'LTeamID', 'Ldays_coaches']

    df = df.merge(TeamCoaches_w,on=['Season','WTeamID'],how='left')
    df = df.merge(TeamCoaches_l,on=['Season','LTeamID'],how='left')

    # add diff features
    df['diff_days_coaches'] = df['Wdays_coaches']-df['Ldays_coaches']

    # merge conferences
    Conferences_w = Conferences.copy()
    Conferences_l = Conferences.copy()

    Conferences_w.columns = ['Season', 'WTeamID', 'WConfAbbrev']
    Conferences_l.columns = ['Season', 'LTeamID', 'LConfAbbrev']

    df = df.merge(Conferences_w,on=['Season','WTeamID'],how='left')
    df = df.merge(Conferences_l,on=['Season','LTeamID'],how='left')

    # add diff features
    df['diff_ConfAbbrev'] = df['WConfAbbrev']-df['LConfAbbrev']

    # save as feather
    to_feature(df, '../feats/f101')

    # save feature name list
    features_json = {'features':df.columns.tolist()}
    to_json(features_json,'../configs/101_all_features.json')

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()