import os
import sys
from astro_ghost.PS1QueryFunctions import getAllPostageStamps
from astro_ghost.TNSQueryFunctions import getTNSSpectra
from astro_ghost.NEDQueryFunctions import getNEDSpectra
from astro_ghost.ghostHelperFunctions import *
from astropy.coordinates import SkyCoord
from astro_ghost.classifier import classify
from astropy import units as u
import pandas as pd
from datetime import datetime
import matplotlib
%matplotlib inline
from astropy.time import Time
import seaborn as sns

users = pd.read_csv("/Users/alexgagliano/Documents/Research/TransientRecommender/YSE_prod_users_20210108.csv")
status = pd.read_csv("/Users/alexgagliano/Documents/Research/TransientRecommender/RecommendationSystem_StatusOnly.csv")

from collections import Counter
Counter(status['id'])

basicInfo = status[['Transient', 'id', 'disc_date', 'followup_create_date', 'followup_created_by']]
basicInfo = basicInfo[basicInfo['disc_date'] == basicInfo['disc_date']]

basicInfo['disc_date'] = [x.replace(" ", "T") for x in basicInfo['disc_date'].values]
basicInfo['disc_date_mjd'] = Time(basicInfo['disc_date'].values.astype(str), format='isot', scale='utc').mjd

basicInfo['followup_create_date'] = [x.replace(" ", "T") for x in basicInfo['followup_create_date'].values]
basicInfo['followup_create_date_mjd'] = Time(basicInfo['followup_create_date'].values.astype(str), format='isot', scale='utc').mjd


#first, get photometry for all transients
wellObserved = basicInfo[basicInfo['id'].isin([8349, 1438, 1720, 12513])]
wellObservedNames = np.unique(wellObserved['Transient'])
wellObservedNames
import requests
MY_USERNAME = 'agagliano'
MY_PASSWORD = 'changeme'
#for transient in np.unique(status['Transient']):
for transient in wellObservedNames:
    url = "https://ziggy.ucolick.org/yse/download_photometry/%s"%transient
    r = requests.get(url, auth=(MY_USERNAME, MY_PASSWORD))
    r.content
    with open("/Users/alexgagliano/Documents/Research/TransientRecommender/TransientPhotometry/SNANA_%s.txt"%transient, 'wb') as outfile:
        outfile.write(r.content)

#then, get a time-series history of each user
userHistories = {}
for user_id in np.unique(basicInfo['followup_created_by']):
    tempDF = basicInfo[basicInfo['followup_created_by'] == user_id]
    userHistories[user_id] = tempDF['followup_create_date_mjd']

mt = ['o', 'd', 's', '^', 'D', '.']
band = [['u', 'up'], ['g', 'gp', 'g-ZTF'], ['r', 'rp', 'r-ZTF'], ['i','ip'], ['z', 'zp'],['y']]
styles = ['--', '-.', ':', 'solid', '-']
for id in np.unique(wellObserved['id']):
    tempDF = wellObserved[wellObserved['id'] == id]
    SN = tempDF['Transient'].values[0]
    photometry = pd.read_csv("/Users/alexgagliano/Documents/Research/TransientRecommender/TransientPhotometry/SNANA_%s.txt"%SN, delim_whitespace=True)
    if len(np.unique(tempDF['followup_created_by'])) < 3:
        continue
    else:
        plt.figure(figsize=(10,7))
        plt.gca().invert_yaxis()
        i = -1
        for user_id in np.unique(tempDF['followup_created_by']):
            i += 1
            tempDF_user = tempDF[tempDF['followup_created_by'] == user_id]
            tempDF_user = tempDF_user.sort_values(by='followup_create_date_mjd')
            ranking = np.ones(len(tempDF_user))
#           ranking[0] = 0 # not interesting upon discovery (yet)
            #phonyDates = np.linspace(tempDF_user['disc_date_mjd'].values[0], tempDF_user['followup_create_date_mjd'].values[0], num=5)
            #phonyRanking = np.zeros(len(phonyDates))
            #date = np.concatenate([phonyDates, tempDF_user['followup_create_date_mjd'].values])
            #ranking = np.concatenate([phonyRanking, ranking])
            date = tempDF_user['followup_create_date_mjd'].values
            #plt.plot(date, ranking, 'o-')
            for time in date:
                plt.axvline(x=time, lw=2, ls=styles[i])
            #merged_rankings = pd.DataFrame({'user_id':np.ones(len(date))*user_id, 'item_id':np.ones(len(date))*id,'rating':ranking, 'timestamp':date, 'user_last_rating_timestamp', 'item_last_rated_timestamp'})
            for j in np.arange(len(band)):
                tempphot = photometry[photometry['FLT'].isin(band[j])]
                plt.plot(np.array(tempphot['MJD'], dtype=float), tempphot['MAG']+j, 'o', c=sns.color_palette("husl", 6)[j])
            #plt.xlim((58090, 58300))

np.unique(status['followup_created_by'])
#user_id  item_id  rating  timestamp user_last_rating_timestamp  item_last_rated_timestamp
users
sns.color_palette("husl", 6)
