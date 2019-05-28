import random
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error

#read pickled dataframe
yum_clean = pd.read_pickle('data/yummly_clean.pkl')
df = pd.read_pickle('data/yummly_ingrX.pkl')
df_int = df.copy().astype(int)

print(df.mean().sort_values(ascending=False))

print(df)

ingredients = list(df.columns)

rand_ingr = random.choice(tuple(ingredients))
print('Random Ingredient: ', rand_ingr)
sample_size = 0.03

rand_ingr = 'onion'
# Set ground truth
grnd = df_int[rand_ingr]

# Generate random sample of rows to mask
nan_idx = df_int[rand_ingr].sample(frac=sample_size).index
# print(df_int.loc[nan_idx, rand_ingr])

# Mask rows
df_int.loc[nan_idx, rand_ingr] = np.NAN
print(df_int.loc[nan_idx, rand_ingr])

# Confirm that randomly selected ingredient is only column with masked values
nan_cols = df_int.columns[df_int.isnull().any()]

# print(df_int[rand_ingr].value_counts().index[0].astype(int))




# Impute masked values using mode of random ingredient distribution
mode = df_int[rand_ingr].value_counts().index[0].astype(int)
print('Mode: ', mode)
df_int.fillna(mode, inplace=True)
# print(df_int.loc[nan_idx, rand_ingr])
df_int[rand_ingr] = df_int[rand_ingr].astype(int)
# print(df_int.loc[nan_idx, rand_ingr])

imp = df_int[rand_ingr]

print(grnd)
print(imp)
mse = mean_squared_error(grnd, imp)
mae = mean_absolute_error(grnd, imp)

print ("MSE for ", rand_ingr, ": \t", mse)
print ("\n MAE for ", rand_ingr, ": \t", mse)


# print(nan_idx)
# print(df_int)
# df_int.to_csv('ingredients_masked.csv')


# yum.loc[nanidx, rand_ingr] = np.NAN
# print(nan_idx)

# imp = SimpleImputer(strategy='most_frequent')
# # print(imp.fit_transform(df_int))
# df_imputed = imp.fit_transform(df_int)
#
# # print(df_int.loc[nan_idx, rand_ingr])
# df_imputed = pd.DataFrame(data=df_imputed, columns=ingredients)
# # print(df_imputed)
# print(df_imputed.loc[nan_idx, rand_ingr])
#
# df_imputed.columns = ingredients
# print(df_imputed)

# print(df_int.isna().sum(axis=0))
# df_int.to_csv('ingredients_imputed.csv')








