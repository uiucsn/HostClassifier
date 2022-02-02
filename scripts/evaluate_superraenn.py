#evaluate superraenn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter

%matplotlib inline
savepath = '/Users/alexgagliano/Documents/Research/TransientRecommender/superraenn/superraenn/plots'

sr_path = '/Users/alexgagliano/Documents/Research/TransientRecommender/superraenn/5k/'
#df_superraenn = pd.read_csv(sr_path + "/superprob_0131_5000.csv",delim_whitespace=True)
#df_superraenn = pd.read_csv(sr_path + "superprob_0131_5000_derivedHost.csv", delim_whitespace=True)
df_superraenn = pd.read_csv(sr_path + "superprob_0131_5000_derivedHost_notweak.csv", delim_whitespace=True)

df_superraenn.index = df_superraenn['EventName']
df_superraenn
del df_superraenn['EventName']

df_superraenn['MaxProb'] = df_superraenn.max(axis=1)
df_superraenn['GuessedClass'] = ''

df_superraenn.loc[df_superraenn['MaxProb'] == df_superraenn['SLSNI'], 'GuessedClass'] = 'SLSNI'
df_superraenn.loc[df_superraenn['MaxProb'] == df_superraenn['SNII'], 'GuessedClass'] = 'SNII'
df_superraenn.loc[df_superraenn['MaxProb'] == df_superraenn['SNIIn'], 'GuessedClass'] = 'SNIIn'
df_superraenn.loc[df_superraenn['MaxProb'] == df_superraenn['SNIa'], 'GuessedClass'] = 'SNIa'
df_superraenn.loc[df_superraenn['MaxProb'] == df_superraenn['SNIb'], 'GuessedClass'] = 'SNIb'
df_superraenn.loc[df_superraenn['MaxProb'] == df_superraenn['SNIc'], 'GuessedClass'] = 'SNIc'

test_truth = pd.read_csv(sr_path + "/sn_5k_list_unblindedtestSet.txt", delim_whitespace=True)
test_truth['EventName'] = test_truth['#CID']
test_truth.index = test_truth['EventName']
del test_truth['EventName']

train_truth = pd.read_csv(sr_path + "/sn_5k_list_unblindedtrainSet.txt", delim_whitespace=True)
train_truth['EventName'] = train_truth['#CID']
train_truth.index = train_truth['EventName']
del train_truth['EventName']

testSet = df_superraenn[df_superraenn.index.isin(test_truth.index)]
trainSet = df_superraenn[df_superraenn.index.isin(train_truth.index)]

trainSet = train_truth.merge(trainSet, on='EventName')
testSet = test_truth.merge(testSet, on='EventName')

accTrain = np.sum(trainSet['GuessedClass'] == trainSet['Type'])/len(trainSet)*100
accTrain
#without host info:
97.11666666666666
#with host info:
64.5625
#with host info
97.2

accTest = np.sum(testSet['GuessedClass'] == testSet['Type'])/len(testSet)*100
accTest
#without host info
71.53333333333333
#with host info
63.63333333333333
#with host info
73.03333333333333

cm_train = confusion_matrix(trainSet['Type'], trainSet['GuessedClass'], normalize='true')
cm_test = confusion_matrix(testSet['Type'], testSet['GuessedClass'], normalize='true')
cm_test_unnorm = confusion_matrix(testSet['Type'], testSet['GuessedClass'])


fig = plt.figure(figsize=(10.0, 8.0), dpi=300) #frameon=false
df_cm = pd.DataFrame(cm_train, columns=np.unique(trainSet['Type']), index = np.unique(trainSet['Type']))
df_cm.index.name = 'True Label'
df_cm.columns.name = 'Predicted Label'
plt.figure(figsize = (10,7))
sns.set(font_scale=2)
g = sns.heatmap(df_cm, cmap="Reds", annot=True, fmt=".2f", annot_kws={"size": 30}, linewidths=1, linecolor='black', cbar_kws={"ticks": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}, vmin=0.29, vmax=0.91)# font size
g.set_xticklabels(g.get_xticklabels(), fontsize = 20)
g.set_yticklabels(g.get_yticklabels(), fontsize = 20)
g.set_title("Training Set, Accuracy = %.2f%%"%accTrain)
plt.savefig(savepath + "/superraenn_trainingSet_5000_hostInfo_notweak.png", dpi=200, bbox_inches='tight')

fig = plt.figure(figsize=(10.0, 8.0), dpi=300) #frameon=false
df_test  = pd.DataFrame(cm_test, columns=np.unique(trainSet['Type']), index = np.unique(trainSet['Type']))
df_test.index.name = 'True Label'
df_test.columns.name = 'Predicted Label'
plt.figure(figsize = (10,7))
sns.set(font_scale=2)
g = sns.heatmap(df_test, cmap="Reds", annot=True, annot_kws={"size": 30}, linewidths=1, linecolor='black', cbar_kws={"ticks": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}, vmin=0.29, vmax=0.91)
g.set_xticklabels(g.get_xticklabels(), fontsize = 20)
g.set_yticklabels(g.get_yticklabels(), fontsize = 20)
g.set_title("Test Set, Accuracy = %.2f%%"%accTest)
plt.savefig(savepath + "/superraenn_testingSet_5000_hostInfo_notweak.png", dpi=200, bbox_inches='tight')

DF_testWrong = pd.DataFrame(Counter(testSet.loc[testSet['GuessedClass'] != testSet['Type'], 'GuessedClass']), index=(0,))
DF_testWrong = DF_testWrong.T

DF_testWrong['Class'] = DF_testWrong[0]

plt.figure(figsize=(10,7))
ax = sns.barplot(x=DF_testWrong.index, y=DF_testWrong['Class'])
plt.ylabel("Guessed Transient")


#checking out photometry I guess...
SNe = pd.read_csv("/Users/alexgagliano/Documents/Research/TransientRecommender/superraenn/sn_bigger_list.txt", delim_whitespace=True)
SNe['ID'] = SNe['#CID']

hostInfo = pd.read_csv("/Users/alexgagliano/Documents/Research/TransientRecommender/superraenn/products/host_info_20k.tar.gz")

for band in 'ugrizY':
    hostInfo.loc[hostInfo['hostmag_%s'%band] == 0, 'hostmag_%s'%band]= np.nan

hostInfo.to_csv("/Users/alexgagliano/Documents/Research/TransientRecommender/superraenn/products/host_info_20k.tar.gz",index=False)

#fullTable:
fullTable = SNe.merge(hostInfo)
fullTable = fullTable[fullTable['Type'] != '-']
fullTable

snClass = np.unique(fullTable['Type'])
plt.figure(figsize=(10,7))
for i in np.arange(len(snClass)):
    tempDF = fullTable[fullTable['Type'] == snClass[i]]
    sns.kdeplot(tempDF['Msol'])
plt.xlabel(r"log10($M_{\odot}$)")
plt.xlim((6, 12))

plt.figure(figsize=(10,7))
for i in np.arange(len(snClass)):
    tempDF = fullTable[fullTable['Type'] == snClass[i]]
    sns.kdeplot(tempDF['SFR'])
plt.xlabel(r"log10($SFR$)")

plt.xlim((-4, 2))
#objIDs = pd.read_csv("/Users/alexgagliano/Documents/Research/TransientRecommender/superraenn/products/host_info_20k.tar.gz")

hostInfo
