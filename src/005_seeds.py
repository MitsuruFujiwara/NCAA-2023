
import numpy as np
import pandas as pd
import sys

from utils import save2pkl, line_notify
from utils import DICT_SEED, DICT_REGION

#==============================================================================
# preprocess seeds
#==============================================================================

def main():

    # load csv
    MTourneySeeds = pd.read_csv('../input/MNCAATourneySeeds.csv')
    WTourneySeeds = pd.read_csv('../input/WNCAATourneySeeds.csv')

    # merge
    TourneySeeds = pd.concat([MTourneySeeds,WTourneySeeds])

    # split seeds
    TourneySeeds['region'] = TourneySeeds['Seed'].apply(lambda x: x[0]).map(DICT_REGION)
    TourneySeeds['seed_in_region'] = TourneySeeds['Seed'].apply(lambda x: x[1:3]).astype(int)
    TourneySeeds['seed_in_region2'] = TourneySeeds['Seed'].apply(lambda x: x[-1]).map(DICT_SEED)

    # drop seed
    TourneySeeds.drop('Seed',axis=1,inplace=True)

    # save pkl
    save2pkl('../feats/seeds.pkl', TourneySeeds)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()