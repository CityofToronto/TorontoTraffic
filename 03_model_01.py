

#data modules
import numpy as np
import pandas as pd

#plotting modules
import matplotlib.pyplot as plt
import seaborn as sns; #sns.set()
import time as tm

#my own modules
import ellipse_functions as ef
import evaluate_model as eva

#Machine Learning: Gaussian Mixture Model for clustering
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA

#set plotting style
plt.style.use('seaborn-white')
palette = sns.color_palette()


# %% user setup

nClusters = 3   #number of clusters to search for
printFigs = True
dataFile = 'data/all/daily_users_all.pkl'  #'data/5M/daily_users_5M_random.pkl'
test_id = 'PCA_GMM_03'

#label e.g. PCA_GMM_02 = 
#   Principal Component Analysis, Gaussian Mixture Model, 2 clusters

        
# %% Load data

#load user-based data frame
start = tm.time()
daily_users = pd.read_pickle(dataFile)
end = tm.time()
print('Done Loading')
print(end-start)


# %% Prepare data

#lose the usernames (because there are some repeats in the \
#daily-hashed IDs, but we want everyone to have a unique ID)
daily_users.reset_index(drop=True,inplace=True)

#don't need the date for the purpose of this
X = daily_users.drop(['date'],axis=1)

#for the model, drop everyone who did only one or two trips
X = X[X.nTrips>2]
X = X.dropna()

#Gimme info
print('Removing users with less than 3 trips')
print('leaves us with this % of data: ')
print(len(X)/len(daily_users)*100)

#find and drop rows with NaNs
inan = X[X.isna().any(axis=1)].index
daily_users.drop(inan,inplace=True)

#It probably makes more sense to wrap the DeltaTs into a 24 hr circle
#should actually be doing this before calculating the median...
ichange = X.DeltaT>12;
X.loc[ichange,'DeltaT'] = 24 - X.loc[ichange,'DeltaT']


# %% instantiate the model

start = tm.time()
model = GMM(n_components = nClusters ,
            covariance_type='full')
end = tm.time()
print(end-start)


# %% Principal Component analysis

print('Doing PCA')
pca = PCA(2) #project into 2 components
Xproj = pca.fit_transform(X)
print('Done PCA')
print(X.shape)
print(Xproj.shape)


# %% draw the clusters on the principal axes

print('Drawing Clusters')
fig, ax = plt.subplots(figsize=(4,4))
sampleN = 1e5;
PlotMe = Xproj[np.random.choice(Xproj.shape[0], int(sampleN), replace=False),:]
ef.plot_gmm(model,PlotMe,ax)
ax.set_xlim(-10, 70)
ax.set_ylim(-30, 30)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

if printFigs:
    plt.savefig('clusters'+'_'+test_id+'.png',dpi=300,bbox_inches="tight")
print('Done Drawing Clusters')

# %% apply model - i.e. do the clustering

print('Start clustering algorithm')
start = tm.time()

#If you're doing PCA
model.fit(Xproj) #do the fit
X['cluster'] = model.predict(Xproj) #get cluster labels

#If you're not doing PCA
#model.fit(X) #do the fit
#X['cluster'] = model.predict(X) #get cluster labels

end = tm.time()
print('End Clustering algorithm')
print(end-start)

# %% print the results

stats = open('stats' + '_' + test_id + '.txt', 'w') 

print('\nNumber of Users in each cluster:', file = stats)
print(X.groupby('cluster')['nTrips'].count(), file = stats)
print(' ', file=stats)

print('Number of Trips in each cluster:', file=stats)
print(X.groupby('cluster')['nTrips'].sum(), file=stats)
print(' ', file=stats)

stats.close()


# %% evaluate model: draw histograms

print('Start drawing histograms')
eva.draw_hists(X, daily_users, printFigs=printFigs, fname_append=test_id, 
               nClusters=nClusters)
print('End drawing histograms')

# %% Add cluster information back to the original dataframe

daily_users['cluster'] = X['cluster'];
daily_users['cluster'].fillna(-1,inplace=True)
daily_users['cluster'] = daily_users['cluster'].astype(int)


# %% Make time series of relative proportions for each group - # of trips

#make a new dataframe to do this
nTrips = pd.DataFrame(columns=['All','LowVol','Commuter','Commercial','Mixed'])

#First column is net sum. Other columns are relative proportions
nTrips['All'] = daily_users.groupby('date')['nTrips'].sum();
#Low Volume Users, percentage of total
nTrips['LowVol'] = daily_users[daily_users['cluster'] == -1 ].groupby(
        'date')['nTrips'].sum() / nTrips['All'];
#Commuter Behaviour Users, percentage of total
nTrips['Commuter'] = daily_users[daily_users['cluster'] == 2].groupby(
        'date')['nTrips'].sum() / nTrips['All'];        
#Commercial Behaviour Users, percentage of total
nTrips['Commercial'] = daily_users[daily_users['cluster'] == 1].groupby(
        'date')['nTrips'].sum() / nTrips['All'];
#Mixed Behaviour Users, percentage of total
nTrips['Mixed'] = daily_users[daily_users['cluster']==0].groupby(
        'date')['nTrips'].sum() / nTrips['All'];
        
        
# %% some of this doesn't make sense. drop the ones with nTrips<50,000

ibad = nTrips['All']<50000;
nTrips.drop(nTrips[ibad].index,inplace=True)


# %% plotting total number of cars on the road

nTrips['All'].plot()
nTrips['All'].rolling(5).median().rolling(5).mean().plot()


# %% Look at results

nTrips['LowVol'].plot()
nTrips['LowVol'].rolling(15).median().plot()

nTrips['Commuter'].plot()
nTrips['Commuter'].rolling(15).median().plot()

nTrips['Commercial'].plot()
nTrips['Commercial'].rolling(15).median().plot()

nTrips['Mixed'].plot()
nTrips['Mixed'].rolling(15).median().plot()

plt.ylim(0,0.6)


# %% Apply rolling median to every column

nTrips = nTrips.rolling(15).median()


# %% Make a stacked plot

fig, ax = plt.subplots()
ax.stackplot(nTrips.index.values, nTrips.drop('All',axis=1).T,zorder=-5)
ax.grid()
#ax.set_axisabove()


# %% Hmmm just do a regular line plot instead

fig, ax = plt.subplots(2,1)
nTrips.drop('All',axis=1).plot(ax=ax[0],color=palette)
ax[0].set_ylim(0,0.45)
ax[0].grid()

nTrips['All'].plot(ax=ax[1])

#okay so it's hard to know whether or not this is a real signal. The number of 
#routes being observed decreased with time (not shown here) and so the number
#of observations also decreased. So it's unclear if the change in the makeup of
#users over time is a real signal or just an artifact of a change in the 
#sampling











































