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

# impute missing values
imp = MICEData(df)
imp.update_all(15)

# Load imputed values into DataFrame and validate there is no missing data
df = pd.DataFrame(imp.data)
print("\n\n df info: ", df.info())
print("\n\n Amount of NAs\n", df.isna().sum())

# save imputed results to csv
imp.data.to_csv('training_with_imputed_15.csv')
