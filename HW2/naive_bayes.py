import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import seaborn as sns

sns.set()

df = pd.read_csv('KaggleV2-May-2016.csv')

# Check for missing values
print(df.info())

# rename columns
df.rename(columns={
    'PatientId': 'patient_id',
    'AppointmentID': 'appointment_id',
    'Gender': 'gender',
    'ScheduledDay': 'scheduled_day',
    'AppointmentDay': 'appointment_day',
    'Age': 'age',
    'Neighbourhood': 'neighborhood',
    'Scholarship': 'scholarship',
    'Hipertension': 'hypertension',
    'Diabetes': 'diabetes',
    'Alcoholism': 'alcoholism',
    'Handcap': 'handicap',
    'SMS_received': 'sms_received',
    'No-show': 'no_show'
}, inplace=True)

# Remove Appointment ID
df.drop(['appointment_id'], axis=1, inplace=True)

# Format Patient ID, Gender, and Neighborhood to a string
df.patient_id = df['patient_id'].apply(lambda patient: str(int(patient)))
df.gender = df['gender'].apply(lambda patient_gender: str(patient_gender))
df.neighborhood = df['neighborhood'].apply(lambda patient_neighborhood: str(patient_neighborhood))

# format dates
df.scheduled_day = pd.to_datetime(df.scheduled_day)
df.appointment_day = pd.to_datetime(df.appointment_day)

# delete row with age < 0
df = df.query('age >= 0')

# Transform no-shows and gender
df.no_show = df['no_show'].map({'Yes': 1, 'No': 0})
df.gender = df['gender'].map({'M': 1, 'F': 0})


# add columns for day of week of appointment and scheduled days
df['appointment_day_of_week'] = df.appointment_day.map(lambda day: day.dayofweek)
df['scheduled_day_of_week'] = df.scheduled_day.map(lambda day: day.dayofweek)


# add column for days elapsed between scheduled date and appointment date
df['appointment_lag'] = df.appointment_day - df.scheduled_day
df.appointment_lag = df.appointment_lag.abs().dt.days

# print(df.query('appointment_lag > 60'))

gender = df.gender
# scheduled_day = df.scheduled_day
# appointment_day = df.appointment_day
appointment_day_of_week = df.appointment_day_of_week
scheduled_day_of_week = df.scheduled_day_of_week
appointment_lag = df.appointment_lag
age = df.age
# neighborhood = df.neighborhood
scholarship = df.scholarship
hypertension = df.hypertension
diabetes = df.diabetes
alcoholism = df.alcoholism
handicap = df.handicap
sms_received = df.sms_received
no_show = df.no_show

# Create subset of df for correlation matrix
model_data = pd.concat([
    gender,
    appointment_day_of_week,
    scheduled_day_of_week,
    appointment_lag,
    age,
    sms_received,
    no_show
], axis=1)

# separate features from labeled test set
X = model_data.loc[:, model_data.columns != 'no_show']
y = model_data.loc[:, model_data.columns == 'no_show']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)


# Fit Naive Bayes model
model = GaussianNB()
model.fit(X_train, np.ravel(y_train))

# Get probability estimates for no-shows
probs = model.predict_proba(X_test)
probs = probs[:, 1]


auc = roc_auc_score(y_test, probs)
print('\n\nArea Under Curve: %.3f' % auc)

# Get false-positive rate, true-positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, probs)

# # Plot the roc curve
plt.plot([0,1], [0,1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.title('Naive Bayes ROC')
plt.show()

# Perform cross-validation
scores = cross_val_score(model, X_train, np.ravel(y_train), cv=10)
print("\n\nCross-validated scores:", scores)


# Predict logistic model test results
y_predicted = model.predict(X_test)
print('\n\nAccuracy of model classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))

# Build confusion matrix
confusion_matrix = confusion_matrix(y_test, y_predicted)
print("\n\n Confusion Matrix:\n", confusion_matrix)

# Build classification report
print("\n\n", classification_report(y_test, y_predicted))