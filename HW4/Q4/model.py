import pandas as pd

import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def xgb_test(X,y):
    print('here')
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=10)
    model = xgb.XGBClassifier()
    print('here2')
    model.fit(X_train, y_train)
    print('here3')
    y_pred = model.predict(X_test)
    print('here4')
    print('First round:', accuracy_score(y_test,y_pred))

def logistic_test(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=10)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('First round:', accuracy_score(y_test,y_pred))
    #tune parameter C
    crange =[0.01,0.1,1,10,100]
    for num in crange:
        model = LogisticRegression(C=num)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print('C=', num, ',score=', accuracy_score(y_test,y_pred))

def decision_tree_test(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('First round:', accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    #read pickled dataframe
    yum_clean = pd.read_pickle('data/yummly_clean.pkl')

    #create a set of all ingredients in the dataframe
    yum_ingredients=set()
    yum_clean['clean ingredients'].map(lambda x: [yum_ingredients.add(i) for i in x])
    print(len(yum_ingredients))
    print('yum ingredients: ', yum_ingredients)
    #create one column for each ingredient, True or False
    yum = yum_clean.copy()
    for item in yum_ingredients:
        yum[item] = yum['clean ingredients'].apply(lambda x: item in x)
    yum_X = yum.drop(yum_clean.columns,axis=1)

    #test various classification models
    decision_tree_test(yum_X, yum['cuisine'])
    xgb_test(yum_X, yum['cuisine'])



