
import numpy as np
import pandas as pd
import sys

from utils import save2pkl, line_notify

#==============================================================================
# preprocess teams
#==============================================================================

def main():

    # load csv
    MTeams = pd.read_csv('../input/MTeams.csv')
    WTeams = pd.read_csv('../input/WTeams.csv')

    # merge
    Teams = pd.concat([MTeams,WTeams])

    # add features
    Teams['diff_D1Season'] = Teams['LastD1Season'] - Teams['FirstD1Season']

    # drop team name
    Teams.drop('TeamName',axis=1,inplace=True)

    # rename columns
    Teams.columns = ['TeamID', 'Teams_FirstD1Season', 'Teams_LastD1Season', 'Teams_diff_D1Season']

    # save pkl
    save2pkl('../feats/teams.pkl', Teams)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()