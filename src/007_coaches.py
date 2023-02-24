
import numpy as np
import pandas as pd
import sys

from utils import save2pkl, line_notify

#==============================================================================
# preprocess coaches
#==============================================================================

def main():

    # load csv
    TeamCoaches = pd.read_csv('../input/MTeamCoaches.csv')

    # add days coaches
    TeamCoaches['days_coaches'] = TeamCoaches['LastDayNum'] - TeamCoaches['FirstDayNum']

    # aggregate
    TeamCoaches = TeamCoaches.groupby(['Season','TeamID']).mean().reset_index()

    # drop unnecessary columns
    TeamCoaches.drop(['FirstDayNum','LastDayNum'],axis=1,inplace=True)

    # save pkl
    save2pkl('../feats/coaches.pkl', TeamCoaches)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()