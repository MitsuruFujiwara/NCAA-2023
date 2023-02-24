
import numpy as np
import pandas as pd
import sys

from utils import save2pkl, line_notify

#==============================================================================
# preprocess massaey ordinals
#==============================================================================

def main():

    # load csv
    MasseyOrdinals = pd.read_csv('../input/MMasseyOrdinals.csv')

    # drop syetem name
    MasseyOrdinals.drop(['SystemName'],axis=1,inplace=True)

    # get stats features
    df_mean = MasseyOrdinals.groupby(['Season','TeamID']).mean()
    df_std = MasseyOrdinals.groupby(['Season','TeamID']).std()
    df_max = MasseyOrdinals.groupby(['Season','TeamID']).max()
    df_min = MasseyOrdinals.groupby(['Season','TeamID']).min()
    df_sum = MasseyOrdinals.groupby(['Season','TeamID']).sum()

    df_mean.columns = [f'{c}_mean' for c in df_mean.columns]
    df_std.columns = [f'{c}_std' for c in df_std.columns]
    df_max.columns = [f'{c}_max' for c in df_max.columns]
    df_min.columns = [f'{c}_min' for c in df_min.columns]
    df_sum.columns = [f'{c}_sum' for c in df_sum.columns]

    df_mean.reset_index(inplace=True)
    df_std.reset_index(inplace=True)
    df_max.reset_index(inplace=True)
    df_min.reset_index(inplace=True)
    df_sum.reset_index(inplace=True)

    df = df_mean.merge(df_std,on=['Season','TeamID'],how='outer')
    df = df.merge(df_max,on=['Season','TeamID'],how='outer')
    df = df.merge(df_min,on=['Season','TeamID'],how='outer')
    df = df.merge(df_sum,on=['Season','TeamID'],how='outer')

    # save pkl
    save2pkl('../feats/massey_ordinals.pkl', df)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()