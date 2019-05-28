import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap

from xgboost import XGBRegressor

# Create dataframe
df = pd.read_csv('mimic_los_subset.csv')

# Separate feature-set from predicted variable
X = df.drop(['actualLOS'], axis=1)
y = df['actualLOS']
y_arr = np.asarray(df['actualLOS'], dtype="|S6")

# Create classifier for feature selection
clf_etc = ExtraTreesClassifier()
clf_etc.fit(X,y_arr)

# Get top n features
n = 40
feat_importances = pd.Series(clf_etc.feature_importances_, index=X.columns)
feature_list = feat_importances.nlargest(n).index.tolist()

# Select top n features for model
df = df[feature_list]
X = df

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=7)

# Set data instances to be explained by shap
num_prediction_rows = 5
data_to_predict = X_train.iloc[0:num_prediction_rows]
data_to_predict_arr = data_to_predict.values.reshape(1,-1)


# Create XGB Model
# xgb = XGBClassifier(silent=False, learning_rate=0.1, n_estimators=5, max_depth=5, subsample=0.7, colsample_bytree=1, gamma=1)
xgb = XGBRegressor()
xgb.fit(X_train, y_train)


# Get XGB predictions on test set
y_pred = xgb.predict(X_test)

# Evaluate XG Boost model
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

# Generate shap explanations
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(data_to_predict)
shap.summary_plot(shap_values, data_to_predict)

# Add predictions to df and save to csv
y_pred = xgb.predict(X_train)
X_train['actualLOS'] = y_pred
X_train.set_index('age', inplace=True)
X_train.to_csv(r'./mimic_xgb_train.csv')

X_test['actualLOS'] = y_test
X_test.set_index('age', inplace=True)
X_test.to_csv(r'./mimic_xgb_test.csv')







