import pandas as pd
import numpy as np
from sklearn import metrics

imp = pd.read_csv('training_with_imputed.csv')
grnd = pd.read_csv('training_groundtruth.csv')

print(imp.info())
print(grnd.info())

print(imp.index)



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

# Validate Delete was successful
print(imp.info())
print(grnd.info())


# not yet working
for column in imp:
    rsme = ((imp[column] - grnd[column]) ** 2).mean() ** .5
    print ("RSME for ", column, ": ", rsme)


