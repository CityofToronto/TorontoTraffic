# TorontoTraffic

In order to plan for sustainable street use in the downtown core, the City of Toronto is trying to understand the effect that hired vehicles have on traffic density. The question is motivated by the recent realization that ride-hailing services are responsible for increases in traffic congestion in other major North American cities like San Francisco and New York, the latter of which recently introduced a cap on the number of ride-sharing vehicles.

Using measurements from passive bluetooth sensors deployed across downtown Toronto and an unsupervised machine learning (clustering) algorithm, I developed a model that distinguishes between road users and labels them as "hired" or "commuter" based on their behavioural patterns. The model quantifies the effects that hired vehicles (including taxis, ride-hailing vehicles, and delivery vehicles) have on Toronto traffic congestion and, going forward, can be used to monitor those effects over time.

Check out https://www.eoas.ubc.ca/~bscheife/index.html for a blog-format overview of the project and results!


## Prerequisites & Data

In addition to the scripts given here, you'll need python (pandas, scikit-learn, matplotlib, seaborn, numpy). You're also going to want the data. A very small subset of the data are included here as a pickle, but you'll want more if you're going to do any analysis yourself. 

The full set of data can be obtained by contacting the City of Toronto. I worked with records of 83 million trips completed across downtown Toronto between Oct 2017 and May 2019. Note that MAC addresses are anonymized with a hash that regenerates every 24 hours. The upshot is that (i) no user can ever be individually identified, and (ii) no individual user may be traced for longer than 24 hours.


## Getting Started

Start with 01_data_wrangling.py. This loads the data, puts it into pandas dataframes, and massages it into a format from which we can start obtaining insight. Specifically, the data are grouped and summarized by UserID at a daily frequency. Note that the anonymized UserIDs are rehashed every 24 hours, so there's no point in grouping at a lower frequency than daily.

Next, you can get some basic insight into the data with 02_stats_plots.py. The most interesting bits here are the climatologies of typical traffic densities which can help you plan your commute to minimize your time sitting in traffic. Scripts are broken up into cells with # %% dividers that can be understood by spyder.

The script 03_model_01.py runs the machine learning model. It first carries out a principal component analysis to reduce feature dimensions from 5 to 2, and then runs a Gaussian Mixture Model to create 3 clusters that have "Commuter", "Commercial", and "Mixed" behavioural patterns. "Low Volume Users" (i.e. people who are definitely commuters since they only show up once or twice per day) are labelled before the algorithm runs.


## Authors

Just me


## Acknowledgments

Thanks to my team leaders and fellow Fellows at Insight Data Science. Props too to the City of Toronto for using data science and machine learning to make informed policy decisions!

