import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.ensemble import ExtraTreesClassifier



df = pd.read_csv('mimic_los_subset.csv')

# Delete rows with negative values
df = df.query('minArterialPressurePast12months >= 0 & edTimeBeforeAdmission >= 0 ')
# print(df.info())
# Separate feature-set from predicted variable
X = df.drop(['actualLOS'], axis=1)
# y = df['actualLOS']
# y = list(df.actualLOS.values)
y = np.asarray(df['actualLOS'], dtype="|S6")
print(y)

# print(df.columns[df.isna().any()])
# print(df.isna().any())


# print(df.query('edTimeBeforeAdmission < 0').count())
# print(df.query('minArterialPressurePast12months < 0 & edTimeBeforeAdmission < 0 ').count())

# # Get correlations
# corrmat = df.corr()
#
# top_features = corrmat.index
# plt.figure(figsize=(20,20))
#
# # Plot Heatmap
# sns.heatmap(df[top_features].corr(), annot=True)
# plt.show()


features = SelectKBest(score_func=chi2, k=10)
fit = features.fit(X,y)
# print("fit scores: ", fit.scores_)
scores = pd.DataFrame(fit.scores_)
print(scores)
columns = pd.DataFrame(X.columns)

feature_scores = pd.concat([scores, columns],axis=1)
feature_scores.columns = ['score', 'feature']
print('\n\n\n')
print(feature_scores.nlargest(20, 'score'))

feature_list = list(feature_scores.nlargest(20, 'score').feature)

# Create correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
print(to_drop)
df.drop(df.columns[to_drop], axis=1)


clf_etc = ExtraTreesClassifier()
clf_etc.fit(X,y)

print(clf_etc.feature_importances_)

# put feat importances into an array
# build corr matrix of feat importances w/ actualLOS
feat_importances = pd.Series(clf_etc.feature_importances_, index=X.columns)


# feat_importances.nlargest(10).plot(kind='barh')
# plt.show()

# scores = pd.DataFrame(clf_etc.feature_importances_)
# columns = pd.DataFrame(X.columns)
# feature_scores = pd.concat([columns, scores], axis=1)
# feature_scores.columns = ['feature', 'score']
# feature_list = feature_scores.nlargest(10, 'score')

# get correlation matrix
feature_list = feat_importances.nlargest(40).index.tolist()
print(feature_list)
df_features = df[feature_list]
df_features['actualLOS'] = df['actualLOS']
print(df_features.info())
print(df.info())



sns.heatmap(df_features.corr())
plt.figure(figsize=(30,30))
plt.show()
