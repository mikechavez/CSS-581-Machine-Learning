import pandas as pd
from sklearn import metrics

imp = pd.read_csv('training_with_imputed.csv')
grnd = pd.read_csv('training_groundtruth.csv')

print(imp.info())
print(grnd.info())

# not yet working
for column in imp:
    rsme = ((imp[column] - grnd[column]) ** 2).mean() ** .5
    print ("RSME for ", column, ": ", rsme)


