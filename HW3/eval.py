import pandas as pd

imp = pd.read_csv('training_with_imputed.csv')
grnd = pd.read_csv('training_groundtruth.csv')

# not yet working : (
for column in imp:
    rsme = ((imp[column] - grnd[column]) ** 2).mean() ** .5
    print ("RSME for ", column, ": ", rsme)