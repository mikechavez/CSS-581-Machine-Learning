import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.interactive(False)

sns.set(color_codes=True)


# read csv into pandas dataframe
raw_data = pd.read_csv('mimic3data-flow-pred.csv')

# copy data
data = raw_data.copy()

# print((data[['lastValueSystolicBPInPast12Months', 'lastValueDiastolicBPInPast12Months']] == '').sum())

# Replace all blank values in data with NaN
data.replace('', np.nan, regex=True)

lastSystolic = data['lastValueSystolicBPInPast12Months']
lastDiastolic = data['lastValueDiastolicBPInPast12Months']

# Replace last systolic values < 20 with NaN
# data.loc[data['lastValueSystolicBPInPast12Months'] < 20, 'lastValueSystolicBPInPast12Months'] = np.nan
#
# # Replace last diastolic values < 20 with NaN
# data.loc[data['lastValueDiastolicBPInPast12Months'] < 20, 'lastValueDiastolicBPInPast12Months'] = np.nan
#
# data['lastValueSystolicBPInPast12Months'] = data['lastValueSystolicBPInPast12Months'].transform(lambda x: x.fillna(x.mean()))
#
# data['lastValueDiastolicBPInPast12Months'] = data['lastValueDiastolicBPInPast12Months'].transform(lambda x: x.fillna(x.mean()))


# Replace erroneous systolic values with NaN
data.loc[lastSystolic < 20, 'lastValueSystolicBPInPast12Months'] = np.nan
data.loc[lastSystolic > 200, 'lastValueSystolicBPInPast12Months'] = np.nan

# Replace erroneous diastolic values with NaN
data.loc[lastDiastolic < 20, 'lastValueDiastolicBPInPast12Months'] = np.nan
data.loc[lastDiastolic > 2000, 'lastValueDiastolicBPInPast12Months'] = np.nan

lastSystolic= lastSystolic.transform(lambda x: x.fillna(x.mean()))

lastDiastolic = lastDiastolic.transform(lambda x: x.fillna(x.mean()))

print(lastSystolic)

print(lastSystolic.describe())

plt.hist(lastSystolic)
plt.show()

plt.hist(lastDiastolic)
plt.show()



