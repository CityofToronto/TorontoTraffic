
# Time to run this script:
    # 2e4 data: 5 s
    # 2e5 data: 50 s
    # 5e6 data: 21 mins
    # all data (8.2e7): 167 mins

# %% Numerical tools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random as rnd


# %% Load Data

#start timer for everything
startscript = time.time()

#save to file?
fname = 'daily_users_all.pkl'
fname_time = 'time_series_all.pkl'
saveFile = True

start = time.time()
traffic = pd.read_csv('data/raw_data/bluetooth_insight.csv') #all data
end = time.time()
print(end-start)


# %% initial data wrangling

#start timer
start = time.time()

#Drop useless columns and rename others
traffic.drop(columns=['cod', 'device_class'],inplace=True)
traffic.rename(columns={'analysis_id':'route', \
                        'measured_time':'travel_time', \
                        'measured_timestamp':'time'}, inplace=True)

#seconds to minutes
traffic.travel_time = traffic.travel_time / 60

#Convert to dates to datetimes and use these as the index
traffic.time = pd.to_datetime(traffic.time)
traffic['Tm'] = traffic.time

#time indeces are very much nonmonotonic, so sort
traffic = traffic.sort_values('time')
traffic.set_index('time', inplace=True)
traffic['doy'] = traffic.index.dayofyear
traffic['date'] = traffic.index.date;
traffic['hour'] = traffic.index.hour;
traffic['weekday']= traffic.index.weekday;

#Weekend yes/no -- 0 = Monday, 6 = Sunday
traffic['weekend'] = np.where(traffic['weekday']>4, True, False)

#After hours yes/no
traffic['afterHrs'] = np.where((traffic['hour']<7)|(traffic['hour']>19),True,False)

#end timer
end = time.time()
print(end-start)


# %% get some basic stats

print('Total number of routes loaded')
print(len(traffic.index))

print('Number of unique users')
print(traffic.user_id.nunique())

print('Number of unique routes')
print(traffic.route.nunique())
print('Routes are')
print(traffic.route.unique())

print('Mean travel time (mins) per route')
print("{:.2f}".format(traffic['travel_time'].mean()))


# %% Group by Date and then by User 

#first create the grouping object
groupObj = traffic.groupby(['date','user_id']);

#total number of trips completed
users_by_day = groupObj['route'].count().to_frame();
users_by_day.rename(columns={'route':'nTrips'},inplace=True)

#total number of unique routes
users_by_day['nRoutes'] = groupObj['route'].nunique();

#median time between trips
users_by_day['DeltaT'] = groupObj['Tm'].unique().apply(np.diff).apply(np.median)

#from python timestamp to decimal hours
users_by_day['DeltaT'] = users_by_day['DeltaT'] / np.timedelta64(1, 'h');

#weekend? mostly after hours?
users_by_day[['Weekend', 'AfterHrs']] = np.floor(
        groupObj['weekend', 'afterHrs'].median())

#Flatten the array back into a simple oneD index
users_by_day.reset_index('date',inplace=True)


# %% end timer everything

endscript = time.time()
print('Total time in seconds to run script:')
print(endscript-startscript)


# %% Save Pickle File

if saveFile:
    users_by_day.to_pickle(fname)
    traffic.to_pickle(fname_time)
    






