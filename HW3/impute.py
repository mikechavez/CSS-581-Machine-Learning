import pandas as pd
import numpy as np
from fancyimpute import IterativeImputer
from statsmodels.imputation.mice import MICE, MICEData, MICEResults

# load csv with missing data
data = pd.read_csv('training_with_missing.csv')

# make a copy of the data
df = data.copy()

# Check for missing data
print(df.info())
print("\n\n Amount of NAs\n", df.isna().sum())

# Add column to count # of missing analytes for each instance
df['missing_count'] = df.isnull().sum(axis=1)


# Transform non-normal variables to normal distributions
# df['PBUN'] = np.log10(df['PBUN'])
# # df['PCL'] = np.log(df['PCL'])
# df['PGLU'] = np.log(df['PGLU'])
# df['PLT'] = np.log(df['PLT'])
# # df['PNA'] = np.log(df['PNA'])

# impute missing values
imp = MICEData(df)
imp.update_all(10)



# Load imputed values into DataFrame and validate there is no missing data
df = pd.DataFrame(imp.data)
print("\n\n df info: ", df.info())
print("\n\n Amount of NAs\n", df.isna().sum())

# save imputed results to csv
imp.data.to_csv('training_with_imputed.csv')
