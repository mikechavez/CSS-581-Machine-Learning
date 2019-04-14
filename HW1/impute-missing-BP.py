import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read csv into pandas dataframe
raw_data = pd.read_csv('mimic3data-flow-pred.csv')

# copy data
data = raw_data.copy()

lastSystolic = data['lastValueSystolicBPInPast12Months']
lastDiastolic = data['lastValueDiastolicBPInPast12Months']

# Replace erroneous systolic values with NaN
data.loc[lastSystolic < 20, 'lastValueSystolicBPInPast12Months'] = np.nan
data.loc[lastSystolic > 200, 'lastValueSystolicBPInPast12Months'] = np.nan

# Replace erroneous diastolic values with NaN
data.loc[lastDiastolic < 20, 'lastValueDiastolicBPInPast12Months'] = np.nan
data.loc[lastDiastolic > 2000, 'lastValueDiastolicBPInPast12Months'] = np.nan

# Impute missing values using the column mean for each type of BP
lastSystolic= lastSystolic.transform(lambda x: x.fillna(x.mean()))
lastDiastolic = lastDiastolic.transform(lambda x: x.fillna(x.mean()))


# Plot the distributions
plt.hist(lastSystolic)
plt.show()

plt.hist(lastDiastolic)
plt.show()



