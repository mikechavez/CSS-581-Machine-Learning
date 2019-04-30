import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc("font", size=14)
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


# Plot correlation matrix
corr = corr_data.corr()
fig, ax = plt.subplots(figsize=(10,10))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
ticks = np.arange(0, len(corr.columns), 1)
ax.set_xticks(ticks)
plt.xticks(rotation=30)
ax.set_yticks(ticks)
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.columns)
plt.show()




# Plot distribution of gender
sns.countplot(x=gender)
plt.title('Gender Distribution')
plt.show()


# # Plot distribution of Appointment Day
sns.countplot(x=appointment_day_of_week)
plt.title('Appointment Day of Week')
plt.show()

appointment_lag_plot = sns.distplot(appointment_lag, hist=True, bins=30);
plt.title('Days Between Scheduled and Appointment Day')
plt.show()

age_plot = sns.distplot(age, hist=True, bins=5);
plt.title('Patient Age')
plt.xlabel('age')
plt.ylabel('Count')
plt.show()

# Plot distribution of no_show
sns.countplot(x=scholarship)
plt.title('Scholarship Participation')
plt.show()


# Plot distribution of hypertension
sns.countplot(x=hypertension)
plt.title('Hypertension Distribution')
plt.show()

# Plot distribution of diabetes
sns.countplot(x=diabetes)
plt.title('Diabetes Distribution')
plt.show()

# Plot distribution of alcoholism
sns.countplot(x=alcoholism)
plt.title('Alcoholism Distribution')
plt.show()

# Plot distribution of handicap
sns.countplot(x=handicap)
plt.title('Handicap Distribution')
plt.show()

# Plot distribution of handicap
sns.countplot(x=sms_received)
plt.title('SMS Received Distribution')
plt.show()


# # Plot distribution of no_show
sns.countplot(x=df['no_show'])
plt.title('Appointment Funnel')
plt.show()