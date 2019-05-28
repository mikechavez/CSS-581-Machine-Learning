import random
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error

#read pickled dataframe
yum_clean = pd.read_pickle('data/yummly_clean.pkl')
df = pd.read_pickle('data/yummly_ingrX.pkl')
df_int = df.copy().astype(int)

# Examine feature means
print(df.mean().sort_values(ascending=False))

# Generate random ingredient
ingredients = list(df.columns)
rand_ingr = random.choice(tuple(ingredients))

# Set sample size
sample_size = 0.05

# Set ground truth for imputation performance evaluation
grnd = df_int[rand_ingr]

# Generate random sample of rows to mask
nan_idx = df_int[rand_ingr].sample(frac=sample_size).index
# print(df_int.loc[nan_idx, rand_ingr])

# Mask rows
df_int.loc[nan_idx, rand_ingr] = np.NAN

# Confirm that randomly selected ingredient is only column with masked values
nan_cols = df_int.columns[df_int.isnull().any()]

# Impute masked values using mode of random ingredient distribution
mode = df_int[rand_ingr].value_counts().index[0].astype(int)
df_int.fillna(mode, inplace=True)

df_int[rand_ingr] = df_int[rand_ingr].astype(int)

imp = df_int[rand_ingr]

mse = mean_squared_error(grnd, imp)
mae = mean_absolute_error(grnd, imp)

print ("MSE for ", rand_ingr, ": \t", mse)
print ("\n MAE for ", rand_ingr, ": \t", mse)









