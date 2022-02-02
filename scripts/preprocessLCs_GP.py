%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import gzip
import pickle
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from astropy import table, coordinates, units as u
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import ephem

simDir = "/Users/alexgagliano/Documents/Research/DESC/tables/SNANASims/10k_Sims/"
prefix = "MLAG_GP_ELASTICC_TEST_LSST_WFD_"

############### pretty plotting routines ########################
sns.set_context("talk",font_scale=1.5)

sns.set_style('white', {'axes.linewidth': 0.5})
plt.rcParams['xtick.major.size'] = 15
plt.rcParams['ytick.major.size'] = 15

plt.rcParams['xtick.minor.size'] = 10
plt.rcParams['ytick.minor.size'] = 10
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.minor.width'] = 2

plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.bottom'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True

plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
## for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
############### end pretty plotting routines ########################
LSST_FILTERS = 'ugrizY'


def read_data(filename):
    """Read data from pickled file to a pandas dataframe"""
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)

    X = to_dataframe(data)
    y = pd.get_dummies(X.type == 0, prefix='SNIax', drop_first=True)
    X = X.drop(columns=['type'])

    return X, y


def to_dataframe(data):
    """Converts from a python dictionary to a pandas dataframe"""
    for idx in data:
        sn = data[idx]
        for filt in LSST_FILTERS:
            sn['mjd_%s' % filt] = np.array(sn[filt]['mjd'])
            sn['fluxcal_%s' % filt] = np.array(sn[filt]['fluxcal'])
            sn['fluxcalerr_%s' % filt] = np.array(sn[filt]['fluxcalerr'])

            #make mag
            sn['mag_%s' % filt] = np.array(-2.5*np.log10(np.abs(sn[filt]['fluxcal'])))+27.5

            sn['snr_%s' % filt] = (sn[filt]['fluxcalerr'] / np.abs(sn[filt]['fluxcal']))


            sn['magerr_%s' % filt] = np.array(1.086 * sn['snr_%s' % filt])
            sn['magerr_%s' % filt][sn['magerr_%s' % filt] > 0.5] = 0.5
            #  find candence

            sn['delta_t_%s' % filt] = [j-i for i, j in zip(sn['mjd_%s' % filt][:-1], sn['mjd_%s' % filt][1:])]
            sn['median_delta_t_%s' % filt] = np.array(np.median(sn['delta_t_%s' % filt]))

            sn['magobs_%s' % filt] = np.array(np.median(sn['delta_t_%s' % filt]))

            del sn[filt]
        sn.update(sn['header'])
        del sn['header']

    return pd.DataFrame.from_dict(data, orient='index')

######################### data loading ##############################
AGN, y = read_data(simDir + "/" + prefix + 'AGN.pkl.gz')
KN_K, y = read_data(simDir + "/" + prefix + 'KN_K17.pkl.gz')
KN_B, y = read_data(simDir + "/" + prefix + 'KN_B19.pkl.gz')
SLSN, y = read_data(simDir + "/" + prefix + 'SLSN-I.pkl.gz')
CC1, y = read_data(simDir + "/" + prefix + 'SNII+HostXT_V19.pkl.gz')
CC2, y = read_data(simDir + "/" + prefix + 'SNII-NMF.pkl.gz')
CC3, y = read_data(simDir + "/" + prefix + 'SNII-Templates.pkl.gz')
SNIIb, y = read_data(simDir + "/" + prefix + 'SNIIb+HostXT_V19.pkl.gz')
SNIIn1, y = read_data(simDir + "/" + prefix + 'SNIIn+HostXT_V19.pkl.gz')
SNIIn2, y = read_data(simDir + "/" + prefix + 'SNIIn-MOSFIT.pkl.gz')
TDE, y = read_data(simDir + "/" + prefix + 'TDE.pkl.gz')
SNIax, y = read_data(simDir + "/" + prefix + 'SNIax.pkl.gz')
SNIa91bg, y = read_data(simDir + "/" + prefix + 'SNIa-91bg.pkl.gz')
SNIb1, y = read_data(simDir + "/" + prefix + 'SNIb+HostXT_V19.pkl.gz')
SNIb2, y = read_data(simDir + "/" + prefix + 'SNIb-Templates.pkl.gz')
SNIc1, y = read_data(simDir + "/" + prefix + 'SNIc+HostXT_V19.pkl.gz')
SNIc2, y = read_data(simDir + "/" + prefix + 'SNIc-Templates.pkl.gz')
SNIcBL, y = read_data(simDir + "/" + prefix + 'SNIcBL+HostXT_V19.pkl.gz')
SNIa, y = read_data(simDir + "/" + prefix + 'SNIa-SALT2.pkl.gz')

KN = pd.concat([KN_K, KN_B], ignore_index=True)
SNIb = pd.concat([SNIb1, SNIb2], ignore_index=True)
SNIc = pd.concat([SNIc1, SNIc2], ignore_index=True)
CC = pd.concat([CC1, CC2, CC3], ignore_index=True)
SNIIn = pd.concat([SNIIn1, SNIIn2], ignore_index=True)

AGN
