
import numpy as np
import pandas as pd
import sys

from utils import save2pkl, line_notify

#==============================================================================
# preprocess seasons
#==============================================================================

def main():

    # load csv
    MSeasons = pd.read_csv('../input/MSeasons.csv')
    WSeasons = pd.read_csv('../input/WSeasons.csv')

    # merge
    Seasons = pd.concat([MSeasons,WSeasons])

    # drop team name
    Seasons.drop('DayZero',axis=1,inplace=True)

    # label encoding
    for r in ['RegionW', 'RegionX', 'RegionY', 'RegionZ']:
        Seasons[r] = Seasons[r].map(Seasons[r].value_counts())

    # save pkl
    save2pkl('../feats/seasons.pkl', Seasons)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()