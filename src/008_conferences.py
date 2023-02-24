
import numpy as np
import pandas as pd
import sys

from utils import save2pkl, line_notify

#==============================================================================
# preprocess conferences
#==============================================================================

def main():

    # load csv
    MConferences = pd.read_csv('../input/MTeamConferences.csv')
    WConferences = pd.read_csv('../input/WTeamConferences.csv')

    # merge
    Conferences = pd.concat([MConferences,WConferences])

    # label encoding
    Conferences['ConfAbbrev'] = Conferences['ConfAbbrev'].map(Conferences['ConfAbbrev'].value_counts())

    # save pkl
    save2pkl('../feats/conferences.pkl', Conferences)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()