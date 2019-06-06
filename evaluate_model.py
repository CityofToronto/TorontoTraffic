

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;
import matplotlib.cm as cm

#set plotting style
sns.set_style('ticks')
sns.set_context('notebook')
sns.set_palette('deep')
palette = sns.color_palette()    
    
def draw_hists(X, daily_users, printFigs, fname_append, nClusters):
    
    
    # %% Pie charts: i. percent of users, ii. percent of trips
    
    fieldname = 'pie'
    fig, ax = plt.subplots(1,2,figsize=(9,4))

    #First do Percent of Users
    
    #number of users by cluster
    pieN = X.groupby('cluster')['nTrips'].count().to_list();
    pieN.append(len(daily_users)-len(X))
    
    #the labels - generalized to nClusters
#    labels = X.groupby('cluster').nTrips.count().index.to_list();
#    labels = [('Cluster ' + str(i)) for i in labels]
#    labels.append('Low Vol.\nUsers ')
    
    #labels - specified
    labels = ['Mixed\nBehaviour','Commercial Behaviour',
              'Commuter Behaviour', 'Low Vol.\nUsers']
    
    #make the pie chart    
    colors = ['#66b3ff', '#ffcc99', '#99ff99', '#ff9999']
    ax[0].pie(pieN, colors = colors, labels=labels, autopct='%1.0f%%', 
            startangle=180)
    
    #draw inner circle
    centre_circle = plt.Circle((0,0),0.25,fc='white')
    ax[0].add_artist(centre_circle)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax[0].axis('equal')
    ax[0].set_title('Users on the Road',fontweight='bold')
    plt.tight_layout()
    
    #Second do pie chart of time spent on roads
    
    #numbers for the pie chart
    pieN = X.groupby('cluster')['nTrips'].sum().to_list();
    pieN.append( daily_users.nTrips.sum() - X.nTrips.sum() )
    
    #the labels for generalized clusters
#    labels = X.groupby('cluster').nTrips.sum().index.to_list();
#    labels = [('Cluster ' + str(i)) for i in labels]
#    labels.append('Low Vol.\nUsers ')
    
    #specified labels
    labels = ['Mixed\nBehaviour','Commercial\nBehaviour',
              'Commuter\nBehaviour', 'Low Vol.\nUsers']
    
    #make the pie chart    
    colors = ['#66b3ff', '#ffcc99', '#99ff99', '#ff9999']
    ax[1].pie(pieN, colors = colors, labels=labels, autopct='%1.0f%%', 
            startangle=115)
    
    #draw inner circle
    centre_circle = plt.Circle((0,0),0.25,fc='white')
    ax[1].add_artist(centre_circle)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax[1].axis('equal')
    ax[1].set_title('Time Spent on the Road',fontweight='bold')
    plt.tight_layout()
    
    #save figure
    if printFigs:
        plt.savefig(fieldname + '_' + fname_append + '.png', 
                    dpi=300, bbox_inches="tight")

        
    # %% the most basic bar chart - number of users in each cluster
    
    fieldname = 'Users'
    
    #After Hours, bar plot
    fig, ax = plt.subplots(figsize=(6,5))
    #could use any field
    counts = X.groupby('cluster')['nTrips'].count() / len(X) * 100; 
    counts.plot(kind='bar',ax=ax, alpha=0.6, color=palette[0:3], rot=0)
    plt.ylabel('% of Users in Cluster')
    plt.ylim(0,100)
    
    #save figure
    if printFigs:
        plt.savefig(fieldname + '_' + fname_append + '.png', 
                    dpi=300, bbox_inches="tight")
    

    # %% another basic bar chart - number of trips in each cluster
    
    fieldname = 'Trips'
    
    #After Hours, bar plot
    fig, ax = plt.subplots(figsize=(6,5))
    #could use any field
    counts = X.groupby('cluster')['nTrips'].sum() / X['nTrips'].sum() * 100; 
    counts.plot(kind='bar',ax=ax, alpha=0.6, color=palette[0:3], rot=0)
    plt.ylabel('% of Trips in Cluster')
    
    #save figure
    if printFigs:
        plt.savefig(fieldname + '_' + fname_append + '.png', 
                    dpi=300, bbox_inches="tight")
    
    
    # %% Number of Trips
    
    fieldname = 'nTrips'
    label = 'Number of Trips / User / day'

    #set up plot
    fig, ax = plt.subplots(nClusters,1,
                           figsize=(3,int(nClusters*2.8)),sharex=True)

    #loop for plotting each cluster in a new panel
    for ii in range(nClusters):
        ax[ii].hist(X[X.cluster==ii][fieldname], color=palette[ii], 
                 bins=np.arange(0,400,15),alpha=0.6,
                 label=('Cluster ' + str(ii)))
        ax[ii].set_yscale('log')
        ax[ii].set_ylim(1e1,8e6)
        ax[ii].legend()
        ax[ii].minorticks_off()

    plt.xlabel(label)
    
    #save figure
    if printFigs:
        plt.savefig(fieldname + '_' + fname_append + '.png', 
                    dpi=300, bbox_inches="tight")

    
    # %% Number of Unique Routes
    
    fieldname = 'nRoutes'
    label = 'Number of Unique Routes / User / day'
    
    #set up plot
    fig, ax = plt.subplots(nClusters,1,
                           figsize=(3,int(nClusters*2.8)),sharex=True)

    #loop for plotting each cluster in a new panel
    for ii in range(nClusters):
        ax[ii].hist(X[X.cluster==ii][fieldname], color=palette[ii], 
                 bins=np.arange(0,120,5),alpha=0.6,
                 label=('Cluster ' + str(ii)))
        ax[ii].set_yscale('log')
        ax[ii].set_ylim(1e0,8e6)
        ax[ii].legend()
        ax[ii].minorticks_off()
        
    #this needs to be outside the loop: bottom axis only
    plt.xlabel(label)
    
    #save figure
    if printFigs:
        plt.savefig(fieldname + '_' + fname_append + '.png', 
                    dpi=300, bbox_inches="tight")
      
    
    # %% median Delta-t
        
    fieldname = 'DeltaT'
    label = 'Median $\Delta$T (hrs) by User'
    
    #set up plot
    fig, ax = plt.subplots(nClusters,1,
                           figsize=(3,int(nClusters*2.8)),sharex=True)

    #loop for plotting each cluster in a new panel
    for ii in range(nClusters):
        ax[ii].hist(X[X.cluster==ii][fieldname], color=palette[ii], 
                 bins=np.arange(0,13,0.5),alpha=0.6,
                 label=('Cluster ' + str(ii)))
        ax[ii].set_yscale('log')
        ax[ii].set_ylim(1e0,8e6)
        ax[ii].legend()
        ax[ii].minorticks_off()
        
    #this needs to be outside the loop: bottom axis only
    plt.xlabel(label)

    
    #save figure
    if printFigs:
        plt.savefig(fieldname + '_' + fname_append + '.png', 
                    dpi=300, bbox_inches="tight")
        
        
        
    # %% Percentage After Hours, Bar Plot
    
    fieldname = 'AfterHrs'
    
    #After Hours, bar plot
    fig, ax = plt.subplots(figsize=(6,5))
    afterHrs = X.groupby('cluster')['AfterHrs'].sum() /  \
        X.groupby('cluster')['AfterHrs'].count() * 100;
    afterHrs.plot(kind='bar',ax=ax, rot=0, alpha=0.6, color=palette[0:3])
    plt.ylabel('% of Users operating After-Hours')
    
    #save figure
    if printFigs:
        plt.savefig(fieldname + '_' + fname_append + '.png', 
                    dpi=300, bbox_inches="tight")
    
    
    
    # %% Percentage on Weekend
    
    fieldname = 'Weekend'
    
    #Weekend, bar plot
    fig, ax = plt.subplots(figsize=(6,5))
    Weekend = X.groupby('cluster')['Weekend'].sum() /  \
        X.groupby('cluster')['Weekend'].count() * 100;
    Weekend.plot(kind='bar',ax=ax, rot=0, color=palette[0:3], alpha=0.6)
    plt.ylabel('% of Users operating Weekends')
    
    #save figure
    if printFigs:
        plt.savefig(fieldname + '_' + fname_append + '.png', 
                    dpi=300, bbox_inches="tight")
    
        
    # %% Make a pair plot of subset of data
    
    sampleN = 1e4;
    pickme = np.random.choice(X.shape[0], int(sampleN), replace=False)
    sns.pairplot(X.iloc[pickme], hue='cluster', plot_kws = 
                 {'alpha': 0.6, 's': 15}, height=1.8)
    
    #save figure
    if printFigs:
        plt.savefig('pairplot' + '_' + fname_append + '.png', 
                    dpi=100, bbox_inches='tight')
    
    






