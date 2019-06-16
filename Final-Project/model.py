import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import shap

def fit_model(X,y,model):
    # Create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7, shuffle=True)
    print('here\n\n',X_test)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return X_test, X_train, y_train, y_test, y_pred, model

def cross_validate_scoring(model, X_train, y_train):
    scoring = {
        'acc': 'accuracy',
        'prec': 'precision',
        'rec': 'recall'
    }
    scores = cross_validate(model, X_train, np.ravel(y_train), scoring=scoring, cv=10, return_train_score=True)
    for metric in scores:
        print("\n\nCross-validated ", metric, ": ", scores[metric])

def generate_roc(clf_type, model, X_test,y_test):
    probs = model.predict_proba(X_test)
    probs = probs[:, 1]
    auc = roc_auc_score(y_test, probs)
    print('\n\nArea Under Curve: %.3f' % auc)
    # Get false-positive rate, true-positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, probs)

    title = clf_type + ' ROC'

    # Plot the roc curve
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    plt.title(title)
    plt.show()

def get_classification_report(y_test, y_pred):
    print("\n\n", classification_report(y_test, y_pred))

def get_confusion_matrix(y_test, y_pred):
    print("\n\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def fit_test_training(X, y, models):

    model_dict = {}
    for clf_type, clf in models.items():
        # Fit the model and get predictions
        X_test, X_train, y_train, y_test, y_pred, model = fit_model(X, y, clf)
        # print('\n\nX_test ', X_test)

        model_dict.update({clf_type: model})

        print('\n\n###################################')
        print('Eval Metrics for ', clf_type, ' model')
        print('###################################')

        # Perform cross-validated scoring, cv=10
        cross_validate_scoring(clf, X_train, y_train)

        # Generate ROC Curve Plot
        generate_roc(clf_type, clf, X_test, y_test)

        # Generate Classification Report
        get_classification_report(y_test, y_pred)

        # Generate Confusion Matrix
        get_confusion_matrix(y_test, y_pred)

        get_shap(X_train, clf_type, clf)
    return X_train, model_dict

def test_model(X, y, models):
    for clf_type, clf in models.items():
        # Get predictions
        y_pred = clf.predict(X)

        print('\n\n###################################')
        print('Eval Metrics for ', clf_type, ' model')
        print('###################################')

        # Generate ROC Curve Plot
        generate_roc(clf_type, clf, X, y)

        # Generate Classification Report
        get_classification_report(y, y_pred)

        # Generate Confusion Matrix
        get_confusion_matrix(y, y_pred)

def get_shap(X_train, clf_type, clf):
    if(clf_type == 'Decision Tree' or clf_type == 'XG Boost'):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_train)
        shap.summary_plot(shap_values, X_train)


#read csv w/ feature-set and target variable
df = pd.read_csv('data/selected_features.csv')
df_test = pd.read_csv('data/test_users.csv')
df_test = df_test[df.columns]

# Separate training feature-set from target variable
X = df.drop(['is_bot'], axis=1)
y = df['is_bot']

# Separate test feature-set from target variable
X_test_users = df_test.drop(['is_bot'], axis=1)
y_test_users = df_test['is_bot']
X_train, X_test, y_train, y_test = train_test_split(X_test_users, y_test_users, test_size=0.99, random_state=7, shuffle=True)

# Generate Classifiers to be compared
xgb_clf = XGBClassifier()
ext_clf = ExtraTreesClassifier(n_estimators=20)
log_clf = LogisticRegression(max_iter=500, solver='lbfgs')
gnb_clf = GaussianNB()

models = {
    "Logistic": log_clf,
    "Gausian Naive Bayes": gnb_clf,
    "Decision Tree": ext_clf,
    "XG Boost": xgb_clf
}

# Use fitted models to test on Cresci 2015 dataset
X_train, fitted_models = fit_test_training(X, y, models)
test_model(X_test, y_test, fitted_models)

# Drop low-contribution features
features = ['is_bot',
            '100x_friends_followers',
            '100x_followers_listed',
            'is_in_a_list',
            'lang_is_En',
            'has_description',
            'has_profile_background_image'
            ]
X = df.drop(features, axis=1)
X_test_users = df_test.drop(features, axis=1)

# Retrain models on reduced feature space
X_train, fitted_models = fit_test_training(X, y, models)

# Re-test Cresci 2015 dataset
X_train, X_test, y_train, y_test = train_test_split(X_test_users, y_test_users, test_size=0.99, random_state=7, shuffle=True)
test_model(X_test, y_test, fitted_models)

