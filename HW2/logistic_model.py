import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv('KaggleV2-May-2016.csv')

print(df)
df.drop(['AppointmentID', 'ScheduledDay'], axis=1, inplace=True)

print(df)
