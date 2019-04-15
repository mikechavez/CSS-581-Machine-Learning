import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read csv into pandas dataframe
raw_data = pd.read_csv('mimic3data-flow-pred.csv')

# make a copy of the data
data = raw_data

# create a subset of the columns labeling chronic conditions
data = data.loc[:, 'isChronicConditionDementia':'hasFrailty']

# remove prefixes
data.columns = data.columns.str.strip().str.replace('isChronicCondition', '').str.replace('has', '')

# plot correlation
corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()