import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

df_train = pd.read_csv('mimic_xgb_train.csv')
df_test = pd.read_csv('mimic_xgb_test.csv')

X_train = df_train.drop(['actualLOS'], axis=1)
y_train = df_train['actualLOS'].to_frame()

X_test = df_test.drop(['actualLOS'], axis=1)
y_test = df_test['actualLOS'].to_frame()

regressor = DecisionTreeRegressor()
regressor = regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

