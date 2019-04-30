import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import seaborn as sns

sns.set()

df = pd.read_csv('KaggleV2-May-2016.csv')

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
appointment_day_of_week = df.appointment_day_of_week
scheduled_day_of_week = df.scheduled_day_of_week
appointment_lag = df.appointment_lag
age = df.age
scholarship = df.scholarship
hypertension = df.hypertension
diabetes = df.diabetes
alcoholism = df.alcoholism
handicap = df.handicap
sms_received = df.sms_received
no_show = df.no_show


# Create subset of df for correlation matrix
corr_data = pd.concat([
    gender,
    appointment_day_of_week,
    scheduled_day_of_week,
    appointment_lag,
    age,
    sms_received,
    no_show
], axis=1)

corr_data.info()



# corr = corr_data.corr()
# fig, ax = plt.subplots(figsize=(10,10))
# colormap = sns.diverging_palette(220, 10, as_cmap=True)
# sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
# ticks = np.arange(0, len(corr.columns), 1)
# ax.set_xticks(ticks)
# plt.xticks(rotation=30)
# ax.set_yticks(ticks)
# ax.set_xticklabels(corr.columns)
# ax.set_yticklabels(corr.columns)
# # plt.xticks(range(ticks=len(corr.columns), labels=corr_data.columns))
# # plt.yticks(range(ticks=len(corr.columns), labels=corr_data.columns))
# plt.show()

# separate features from labeled test set
X = corr_data.loc[:, corr_data.columns != 'no_show']
y = corr_data.loc[:, corr_data.columns == 'no_show']

os = SMOTE(random_state=0)


# # Create over_sampling data
# os_data_X, os_data_y = os.fit_sample(X,np.ravel(y))
#
# # Put oversampled data in to dataframes
# os_data_X = pd.DataFrame(data=os_data_X, columns=X.columns)
# os_data_y = pd.DataFrame(data=os_data_y, columns=y.columns)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)





# # Validate oversampling
# print("\n\nlength of oversampled data is ",len(os_data_X))
# print("Number of people that showed up in oversampled data",len(os_data_y[os_data_y['no_show'] == 0]))
# print("Number of no-shows",len(os_data_y[os_data_y['no_show'] == 1]))
# print("Proportion of people that showed-up data in oversampled data is ",len(os_data_y[os_data_y['no_show'] == 0])/len(os_data_X))
# print("Proportion of no-shows in oversampled data is ",len(os_data_y[os_data_y['no_show'] == 1])/len(os_data_X))
#
# # Change shape of y to 1d array for model fit
# os_data_y = np.ravel(os_data_y)

# Fit logistic model
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, np.ravel(y_train))

# Get probability estimates for no-shows
probs = model.predict_proba(X_test)
probs = probs[:, 1]


auc = roc_auc_score(y_test, probs)
print('\n\nArea Under Curve: %.3f' % auc)

# Get false-positive rate, true-positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, probs)

# Plot the roc curve
plt.plot([0,1], [0,1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.title('Logistic ROC')
plt.show()

# Perform cross-validation
scores = cross_val_score(model, X_train, np.ravel(y_train), cv=10)
print("\n\nCross-validated scores:", scores)

# Fit model after cross-validation
model.fit(X_train, np.ravel(y_train))

# # Get predictions from cross-validation
# predictions = cross_val_predict(logistic_model, X_train, np.ravel(y_train), cv=10)
# print("\n\nCross-validation predictions", predictions)

# Predict logistic model test results
y_predicted = model.predict(X_test)
print('\n\nAccuracy of logistic regression classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))

# Build confusion matrix
confusion_matrix = confusion_matrix(y_test, y_predicted)
print("\n\n Confusion Matrix:\n", confusion_matrix)

# Build classification report
print(classification_report(y_test, y_predicted))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
#
# f, ax = plt.subplots(figsize=(12,9))
# sns.heatmap(corr_data, vmin=1, vmax=1)
# plt.show()



# Plot distribution of gender
# sns.countplot(x=gender)
# plt.title('Gender Distribution')
# plt.show()


# # Plot distribution of Appointment Day
# sns.countplot(x=appointment_day_of_week)
# plt.title('Appointment Day of Week')
# plt.show()

# appointment_lag_plot = sns.distplot(appointment_lag, hist=True, bins=30);
# plt.title('Days Between Scheduled and Appointment Day')
# plt.show()

# age_plot = sns.distplot(age, hist=True, bins=5);
# plt.title('Patient Age')
# plt.xlabel('age')
# plt.ylabel('Count')
# plt.show()

# Plot distribution of no_show
# sns.countplot(x=scholarship)
# plt.title('Scholarship Participation')
# plt.show()


# Plot distribution of hypertension
# sns.countplot(x=hypertension)
# plt.title('Hypertension Distribution')
# plt.show()

# Plot distribution of diabetes
# sns.countplot(x=diabetes)
# plt.title('Diabetes Distribution')
# plt.show()

# Plot distribution of alcoholism
# sns.countplot(x=alcoholism)
# plt.title('Alcoholism Distribution')
# plt.show()

# Plot distribution of handicap
# sns.countplot(x=handicap)
# plt.title('Handicap Distribution')
# plt.show()

# Plot distribution of handicap
# sns.countplot(x=sms_received)
# plt.title('SMS Received Distribution')
# plt.show()


# # Plot distribution of no_show
# sns.countplot(x=df['no_show'])
# plt.title('Appointment Funnel')
# plt.show()

# cols = [
#     'no_show',
#     # 'gender',
#     # 'scheduled_day',
#     # 'appointment_day',
#     # 'appointment_day_of_week',
#     # 'scheduled_day_of_week',
#     # 'appointment_lag',
#     # 'age',
#     # 'neighborhood',
#     # 'scholarship',
#     # 'hypertension',
#     # 'diabetes',
#     # 'alcoholism',
#     # 'handicap',
#     # 'sms_received'
# ]









# sns.pairplot(df[cols], height = 2.5)
# plt.show()

# plt.scatter(age, no_show, color = 'C0')
# plt.xlabel('age', fontsize = 20)
# plt.ylabel('No-show', fontsize = 20)
# plt.show()