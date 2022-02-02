###recommender system initial data analysis
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import json
import feets
from scipy import stats
import feets.preprocess

dirpath = "/Users/alexgagliano/Documents/Research/TransientRecommender/"

logging = pd.read_csv(dirpath + "RecommendationSystem_DjangoLog.csv")
logging['changes'] = [json.loads(x) for x in logging['changes'].values]

temp_status = pd.read_csv(dirpath + "RecommendationSystem_StatusOnly.csv")
interesting = temp_status[temp_status['TransientStatus'].isin(['Following', 'FollowupFinished', 'FollowupRequested', 'Interesting', 'NeedsTemplate'])]

interesting
Counter(interesting['TransientStatus'])

#13896 total transients in YSE-PZs

DF_status = []
DF_modby = []

for idx, row in logging.iterrows():
    if list(row['changes'].keys())[0] in (['status']):
        DF_status.append(logging.iloc[logging.index == idx])
    elif list(row['changes'].keys())[0] in (['modified_by']):
        DF_modby.append(logging.iloc[logging.index == idx])


DF_status = pd.concat(DF_status)
DF_modby = pd.concat(DF_modby)

DF_status['Status_Start'] = np.nan
DF_status['Status_End'] = np.nan

for idx, row in DF_status.iterrows():
     DF_status.loc[DF_status.index == idx, 'Status_Start'] = list(row['changes'].values())[0][0]
     DF_status.loc[DF_status.index == idx, 'Status_End'] = list(row['changes'].values())[0][1]

del DF_status['changes']
del DF_status['additional_data']

np.unique(DF_status['Status_End'])

IgnoreList = DF_status.loc[DF_status['Status_End'] == 'Ignore', 'object_repr'].values

IgnoreSet = temp_status[temp_status['Transient'].isin(IgnoreList)]
IgnoreSet_Final = IgnoreSet[IgnoreSet['TransientStatus'] == 'Ignore']

fullDF = pd.concat([IgnoreSet_Final, interesting])
fullDF.to_csv("/Users/alexgagliano/Documents/Research/TransientRecommender/PreliminaryTransientStatusList.csv",index=False)

fullDF = pd.read_csv("/Users/alexgagliano/Documents/Research/TransientRecommender/PreliminaryTransientStatusList.csv")

all_photometry = pd.read_csv("/Users/alexgagliano/Documents/Research/TransientRecommender/Photometry_and_Host_Galaxy_Information_of_Interesting_and_Ignored_Transients.csv")

photometry_of_tagged = all_photometry[all_photometry['name'].isin(fullDF['Transient'])]
fullDF['name'] = fullDF['Transient']
comb = fullDF.merge(photometry_of_tagged, on='name')
comb.to_csv("/Users/alexgagliano/Documents/Research/TransientRecommender/TaggedTransientData_0125.tar.gz",index=False)

transients = np.unique(comb['Transient'])

comb.columns.values

phot = comb[['Transient', 'obs_date', 'mag', 'mag_err', 'filter']]




for transient in transients:
    # We synchronize the data
    df_transient = comb[comb['Transient'] == transient]
    df_phot = df_transient[['obs_date', 'mag', 'mag_err', 'filter']]

    df_phot_g = df_phot[df_phot['filter']== 'g']
    df_phot_r = df_phot[df_phot['filter']== 'r']
    df_phot_i = df_phot[df_phot['filter']== 'i']
    df_phot_z = df_phot[df_phot['filter']== 'z']


    for df in [df_phot_g, df_phot_r]:#, df_phot_i, df_phot_z]:
        df_set = []
        time = []
        mag = []
        err = []
        if len(df) > 3:
            tempTime = df_phot['obs_date'].values

            d = [pd.to_datetime(x) for x in tempTime]#[0]
            d = [x.value for x in d]
            time.append(d)

            mag.append(df_phot['mag'].values)
            err.append(df_phot['mag_err'].values)
    if len(mag) == 2:
        print("got both!")
        atime, atime2, amag, amag2, aerror, aerror2  = feets.preprocess.align(
            time[0], time[1], mag[0], mag[1], err[0], err[1])

        lc = [time, mag, error,
          mag2, atime, amag, amag2,
          aerror, aerror2]

        fs = feets.FeatureSpace()
        features, values = fs.extract(*lc)
        as_table(features, values)
    else:
        print("only got one.")
        lc = [time[0], mag[0], err[0]]
        fs = feets.FeatureSpace(data=['magnitude','time', 'error'])
        features, values = fs.extract(*lc)
        as_table(features, values)
    break

#only been logging since 2021-10-20
DF_status[DF_status['Status_Start'] == '']
len(DF_status)
Counter(DF_status['Status_End'])
'Following': 15,
         'FollowupFinished': 2,
         'FollowupRequested': 218,
         'Ignore': 8854,
         'Watch': 1241,
         'Interesting': 10,
         'New': 13,
         'Unknown': 2

Counter(DF_status['Status_Start'])
Counter({'FollowupFinished': 2,
         'Following': 8,
         'Watch': 8168,
         'New': 2066,
         'Interesting': 35,
         'Ignore': 51,
         'FollowupRequested': 23,
         'Unknown': 2})

#9836 transients with timestamped transient changes!
from dateutil import parser
DF_modby['date'] = [str(parser.parse(x).date()) for x in DF_modby['timestamp'].values]  # datetime.datetime(1999, 8, 28, 0, 0)
DF_modby['time'] = [str(parser.parse(x).time()) for x in DF_modby['timestamp'].values]  # datetime.datetime(1999, 8, 28, 0, 0)

DF_status['date'] = [str(parser.parse(x).date()) for x in DF_status['timestamp'].values]  # datetime.datetime(1999, 8, 28, 0, 0)
DF_status['time'] = [str(parser.parse(x).time()) for x in DF_status['timestamp'].values]  # datetime.datetime(1999, 8, 28, 0, 0)

DF_status

DF_modby
DF_modby.merge(DF_status, on=['date', 'object_repr'])
