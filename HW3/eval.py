import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

imp = pd.read_csv('training_with_imputed.csv')
grnd = pd.read_csv('training_groundtruth.csv')
train = pd.read_csv('training_with_missing.csv')

# Get indices with missing data
indices_missing_grnd = pd.isnull(grnd).any(1).nonzero()[0]
indices_missing_train = pd.isnull(train).any(1).nonzero()[0]

# Check shape of np array
print(np.shape(indices_missing_grnd))

# Convert NP array to PD DataFrame
indices_missing_grnd = pd.DataFrame(data=indices_missing_grnd.flatten(), columns=['eval_rows'])
indices_missing_train = pd.DataFrame(data=indices_missing_train.flatten(), columns=['eval_rows'])

# Get eval rows
eval_rows = pd.merge(indices_missing_train, indices_missing_grnd, on='eval_rows', how='left', indicator=True).query('_merge == "left_only"').drop('_merge', 1)

# Remove all rows except the ones we want to evaluate from each dataframe
imp = imp[imp.index.isin(eval_rows['eval_rows'])]
grnd = grnd[grnd.index.isin(eval_rows['eval_rows'])]
train = train[train.index.isin(eval_rows['eval_rows'])]

# Confirm indices for evaluation
print(imp)
print(grnd)
print(train)

# Validate Delete was successful
print(imp.info())
print(grnd.info())



print('\n')

# Calculate nRMSE for each analyte manually
for column in imp:
    n = imp.shape[0]
    max = grnd[column].max()
    min = grnd[column].min()

    rmse = round((((imp[column] - grnd[column]) / (max-min)) ** 2).mean() ** .5, 7)
    print ("RSME for ", column, ": \t\t", rmse)

print("\n\n\n")

# Calculate nRMSE for each analyte using sklearn.metrics
for column in imp:
    max = grnd[column].max()
    min = grnd[column].min()
    ground_truth = grnd[column]
    imputed_data = imp[column]
    # print("ground truth: ", ground_truth)
    # print("imputed: ", imputed_data)

    mse = (mean_squared_error(ground_truth, imputed_data))
    rmse = round((mse / (max-min)) ** .5, 7)

    print ("RSME for ", column, ": \t\t", rmse)


