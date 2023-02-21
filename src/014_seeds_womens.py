
import numpy as np
import pandas as pd
import sys

from utils import save2pkl, line_notify
from utils import WBASE_DIR, DICT_SEED, DICT_REGION

#==============================================================================
# preprocess seeds womens
#==============================================================================

def main():

    # load csv
    TourneySeeds = pd.read_csv(f'{WBASE_DIR}/WNCAATourneySeeds.csv')

    # split seeds
    TourneySeeds['region'] = TourneySeeds['Seed'].apply(lambda x: x[0]).map(DICT_REGION)
    TourneySeeds['seed_in_region'] = TourneySeeds['Seed'].apply(lambda x: x[1:3]).astype(int)
    TourneySeeds['seed_in_region2'] = TourneySeeds['Seed'].apply(lambda x: x[-1]).map(DICT_SEED)

    # drop seed
    TourneySeeds.drop('Seed',axis=1,inplace=True)

    # save pkl
    save2pkl('../feats/seeds_womens.pkl', TourneySeeds)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()