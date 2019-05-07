import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

imp = pd.read_csv('training_with_imputed.csv')
grnd = pd.read_csv('training_groundtruth.csv')

print(imp.info())
print(grnd.info())

# Get indices with missing data in ground truth
indices_missing = pd.isnull(grnd).any(1).nonzero()[0]

# Check shape of np array
print(np.shape(indices_missing))

# Convert NP array to PD DataFrame
indices_missing = pd.DataFrame(data=indices_missing.flatten(),columns=['missing_rows'])
print(indices_missing)

# Delete rows with missing data
imp = imp.drop(indices_missing['missing_rows'])
grnd = grnd.drop(indices_missing['missing_rows'])

imp.to_csv(r'./imputed_removed_rows.csv', index=None, header=True)
grnd.to_csv(r'./training_groundtruth_removed_rows.csv', index=None, header=True)



# Validate Delete was successful
print(imp.info())
print(grnd.info())




for column in imp:
    n = imp.shape[0]
    max = grnd[column].max()
    min = grnd[column].min()

    rmse = (((imp[column] - grnd[column]) / (max-min)) ** 2).mean() ** .5
    print ("RSME for ", column, ": ", rmse)

print("\n\n\n")

for column in imp:
    max = grnd[column].max()
    min = grnd[column].min()
    ground_truth = grnd[column]
    imputed_data = imp[column]
    # print("ground truth: ", ground_truth)
    # print("imputed: ", imputed_data)

    mse = (mean_squared_error(ground_truth, imputed_data))
    rmse = (mse / (max-min)) ** .5

    print ("RSME for ", column, ": ", rmse)


