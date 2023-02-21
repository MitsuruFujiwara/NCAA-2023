
import numpy as np
import pandas as pd
import sys

from utils import save2pkl, line_notify
from utils import MBASE_DIR

#==============================================================================
# preprocess seasons mens
#==============================================================================

def main():

    # load csv
    Seasons = pd.read_csv(f'{MBASE_DIR}/MSeasons.csv')

    # drop team name
    Seasons.drop('DayZero',axis=1,inplace=True)

    # label encoding
    for r in ['RegionW', 'RegionX', 'RegionY', 'RegionZ']:
        Seasons[r] = Seasons[r].map(Seasons[r].value_counts())

    # save pkl
    save2pkl('../feats/seasons_mens.pkl', Seasons)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()