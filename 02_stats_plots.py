

# %% Front Matter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import seaborn as sns; sns.set(); 

#set plotting style
sns.set_style('ticks')
sns.set_context('notebook')
sns.set_palette('deep')
palette = sns.color_palette()

#Load data
start = time.time()
traffic = pd.read_pickle('data/all/time_series_all.pkl');
daily_users = pd.read_pickle('data/all/daily_users_all.pkl');
end = time.time()
print(end-start)



# %% get some basic stats & play with python syntax

print('Total number of trips loaded')
print(len(traffic.index))

print('Number of unique user IDs (may be some overlap)')
print(traffic.user_id.nunique())

print('Number of unique routes')
print(traffic.route.nunique())
print('Routes are')
print(traffic.route.unique())

print('Mean travel time (mins) per route')
print("{:.2f}".format(traffic['travel_time'].mean()))



# %% look at raw data first

#Travel Time
fig, ax = plt.subplots();
traffic['travel_time'].hist(ax=ax, bins=np.arange(0,30,2))
plt.xlabel('Travel Time (mins)')
plt.ylabel('Trips')
plt.yscale('log')

#which routes are used most commonly?
fig, ax = plt.subplots()
count = traffic.groupby('route').size()
count.plot(ax=ax)
plt.xlabel('Route Number')
plt.ylabel('Count')

#group by weekday
fig, ax = plt.subplots()
count = traffic.groupby('weekday').size()
count.plot(kind='bar')

#group by hour of day
fig, ax = plt.subplots()
count = traffic.groupby('hour').size()
count.plot(kind='bar')


# %% Time series

#Time series of total trips recorded (daily)
fig, ax = plt.subplots(2,1,sharex='col')
counts_by_day = daily_users.groupby('date')['nTrips'].sum();
ax[0].plot(counts_by_day)
ax[0].set_title('Daily Recorded Trips')
ax[0].set_ylabel('N Trips')
#Add number of unique users
ax[0].plot(daily_users.groupby('date').size())

#Add number of unique routes recorded (daily)
uroutes_by_day = traffic.groupby('date')['route'].nunique()
ax[1].plot(uroutes_by_day)
ax[1].set_ylabel('N Route IDs')


# %% Climatology Weekdays & Weekends

#make the data structure
trips15 = traffic.resample('15T')['user_id'].count()

#setup axes for figure
fig, ax = plt.subplots(1,2,sharey=True, figsize=(9,3.5))

#do it for weekdays only
iweekday = trips15.index.weekday<5;
climate_mean = trips15[iweekday].groupby(trips15[iweekday].index.time).mean()
climate_std = trips15[iweekday].groupby(trips15[iweekday].index.time).std() 

#plot weekday climatology
ax[0].fill_between(climate_mean.index, climate_mean-climate_std, 
  climate_mean+climate_std, color='gray', alpha=0.2)

#plot line
climate_mean.plot(ax=ax[0],lw=2.5)
ax[0].set_ylabel('N / 15 mins')
ax[0].set_xlabel('Time of Day')
ax[0].set_title('Weekdays')

#adjust xticks
ax[0].set_xticks(climate_mean.index[0::16])
ax[0].set_xlim(min(climate_mean.index), max(climate_mean.index))

#set x axis ticks
fig.autofmt_xdate()


#repeat, but do it for weekends only
iweekday = trips15.index.weekday>4;
climate_mean = trips15[iweekday].groupby(trips15[iweekday].index.time).mean()
climate_std = trips15[iweekday].groupby(trips15[iweekday].index.time).std() 

#plot weekend climatology
ax[1].fill_between(climate_mean.index, climate_mean-climate_std, 
  climate_mean+climate_std, color='gray', alpha=0.2)

#plot line
climate_mean.plot(ax=ax[1],lw=2.5)
ax[1].set_ylabel('N / 15 mins')
ax[1].set_xlabel('Time of Day')
ax[1].set_title('Weekend')

#adjust xticks
ax[1].set_xticks(climate_mean.index[0::16])
ax[1].set_xlim(min(climate_mean.index), max(climate_mean.index))

#set x axis ticks
fig.autofmt_xdate()

#save this
plt.savefig('climatologies_traffic.png',dpi=300)


# %% do a climatology for the week
counts_by_day = traffic.resample('D')['route'].count()
climate_mean = counts_by_day.groupby(counts_by_day.index.weekday).mean()
climate_std = counts_by_day.groupby(counts_by_day.index.weekday).std()
#plotting
fig, ax = plt.subplots()
climate_mean.plot(lw=3)
plt.plot(climate_mean+climate_std, color = 'r')
plt.plot(climate_mean-climate_std, color = 'r')
plt.ylabel('N / 15 mins')
plt.xlabel('Time of Day')
plt.title('Number of Users per Day')
plt.xlabel('Day of the Week (0=Mon)')


# %% Histograms individually

#number of trips per user in a day - log y axis
fig, ax = plt.subplots()
daily_users.nTrips.plot(kind='hist',ax=ax,bins=np.arange(0,200,7.5))
ax.set_yscale('log')
plt.xlabel('Number of Trips / User / day')
plt.savefig('nTrips.png', dpi=300, bbox_inches="tight")

#number of routes per user per day
fig, ax = plt.subplots()
daily_users.nRoutes.plot(kind='hist',ax=ax,bins=np.arange(0,150,5))
ax.set_yscale('log')
plt.xlabel('Number of Unique Routes / User / day')
plt.savefig('nRoutes.png', dpi=300, bbox_inches="tight")


# %% redo above histograms, together and make it look nice

fig, ax = plt.subplots(1,2,figsize=(9,3.5),sharey=True)
#first plot number of Trips
daily_users['nTrips'].plot(kind='hist', ax=ax[0], 
           bins=np.arange(0,200,10), alpha=0.8, color=palette[7])
ax[0].set_yscale('log')
ax[0].set_xlabel('Number of Trips / User / day')
ax[0].minorticks_off()
ax[0].set_ylabel('')
#then plot number of Unique Routes
daily_users['nRoutes'].plot(kind='hist', ax=ax[1], 
           bins=np.arange(0,150,7), alpha=0.8, color=palette[7])
ax[1].set_yscale('log')
ax[1].set_xlabel('Number of Unique Routes / User / day')
ax[1].minorticks_off()

#Save me
plt.savefig('nTrips_nRoutes.png', dpi=300, bbox_inches="tight")


# %% Histograms of DeltaT and After Hours

#delta T, log y axis
fig, ax = plt.subplots()
daily_users.DeltaT.plot(kind='hist',ax=ax, bins = range(25))
ax.set_yscale('log')
plt.xlabel('median $\Delta$t (hrs)')
plt.savefig('deltaT.png', dpi=300, bbox_inches="tight")

#After Hours, bar plot
fig, ax = plt.subplots()
afterHrs = pd.Series({'After Hrs':np.sum(daily_users.AfterHrs==1), 
            'Work Hrs':np.sum(daily_users.AfterHrs==0)},name='count');
afterHrs.plot(kind='bar',ax=ax)
plt.savefig('AfterHrs.png', dpi=300, bbox_inches="tight")





























# %% ######################### OLD STUFF ##########################


## %% Power spectrum
##Hmmm this isn't working right now. Probably need to play with the overlaps of
##   the FFTs to get this to work. Maybe come back to it later
#
#import scipy.signal as sp
#fr, spec = sp.welch(trips15, fs=0.25)
#fig, ax = plt.subplots()
#ax.set_yscale('log')
#ax.set_xscale('log')
#plt.plot(fr,spec)


##%% Let's look from the user's perspective
#
##list of all unique users
#unique_users = traffic.user_id.unique()
##number of routes completed by each user
#nroutes_user = traffic.user_id.value_counts()
#nroutes_user = nroutes_user.sort_index()
##number of days on which each user was active
#ndays_user = traffic.groupby('user_id')['date'].nunique()
#ndays_user = ndays_user.sort_index()

## %% does this make sense?
#
##plot histogram of number of days that user was active
#fig,ax = plt.subplots()
#ndays_user.plot(kind='hist',bins=np.arange(0.5,20.5,1))
#plt.xticks(np.arange(1,20))
#plt.xlabel('Number of Days a User was recorded')
#
##average number of records/day for active users
#mean_routes_per_day = nroutes_user / ndays_user
#
##plot number of records / user
#fig, ax = plt.subplots()
#nroutes_user.plot(kind='hist',bins=np.arange(0.5,15.5,1))
#plt.xlabel('Avg. number of records by a User / day')
#
#
## %% sanity check - what times of day am I sampling?
#
##histogram of hour
#traffic['hour'].plot(kind='hist')
#plt.xlabel('Hour of the Day')

