
import numpy as np
import pandas as pd
import sys

from utils import save2pkl, line_notify
from utils import MBASE_DIR

#==============================================================================
# preprocess conferences mens
#==============================================================================

def main():

    # load csv
    Conferences = pd.read_csv(f'{MBASE_DIR}/MTeamConferences.csv')

    # label encoding
    Conferences['ConfAbbrev'] = Conferences['ConfAbbrev'].map(Conferences['ConfAbbrev'].value_counts())

    # save pkl
    save2pkl('../feats/conferences_mens.pkl', Conferences)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()