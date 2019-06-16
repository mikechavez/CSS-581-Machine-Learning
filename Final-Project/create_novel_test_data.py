import pandas as pd
import numpy as np

# Import data for genuine users
df_e13 = pd.read_csv('data/cresci-2015/e13_users.csv')
df_tfp = pd.read_csv('data/cresci-2015/tfp_users.csv')

# Import data for bot users
df_fsf = pd.read_csv('data/cresci-2015/fsf_users.csv')
df_int = pd.read_csv('data/cresci-2015/int_users.csv')
df_twt = pd.read_csv('data/cresci-2015/twt_users.csv')

# Label datasets
df_e13['is_bot'] = 0
df_tfp['is_bot'] = 0
df_fsf['is_bot'] = 1
df_int['is_bot'] = 1
df_twt['is_bot'] = 1

# Concatenate into single dataframe
df = pd.concat([df_e13, df_tfp, df_fsf, df_int, df_twt])

# Fill nulls
df['geo_enabled'] = df['geo_enabled'].fillna(value=0)
df['profile_use_background_image'] = df['profile_use_background_image'].fillna(value=0)

# 20X Friends-to-followers ratio
df['friends_followers_ratio'] = df['friends_count']/df['followers_count']
df.replace([np.Inf, -np.Inf], 0, inplace=True)
df['20x_friends_followers'] = df['friends_followers_ratio'] >= 20
df['20x_friends_followers'] = df['20x_friends_followers'].astype(int)

# 100X Friends-to-followers ratio
df['100x_friends_followers'] = df['friends_followers_ratio'] >= 100
df['100x_friends_followers'] = df['100x_friends_followers'].astype(int)

# 2X Followers-to-listed ratio
df['followers_listed_ratio'] = df['followers_count']/df['listed_count']
df.replace([np.NaN, np.Inf], 0, inplace=True)
df['100x_followers_listed'] = df['followers_listed_ratio'] >= 100
df['100x_followers_listed'] = df['100x_followers_listed'].astype(int)

# Has been in a list
df['is_in_a_list'] = df['listed_count'] >= 1
df['is_in_a_list'] = df['is_in_a_list'].astype(int)

# Selected Language is 'En'
df['lang_is_En'] = df['lang'] == 'en'
df['lang_is_En'] = df['lang_is_En'].astype(int)

# Has never favorited a tweet
df['has_never_favorited'] = df['favourites_count'] == 0
df['has_never_favorited'] = df['has_never_favorited'].astype(int)

# 5X Followers-to-tweets ratio
df['followers_tweets_ratio'] = df['followers_count']/df['statuses_count']
df.replace([np.NaN, np.Inf], 0, inplace=True)
df['5x_followers_tweets'] = df['followers_tweets_ratio'] >= 5
df['5x_followers_tweets'] = df['5x_followers_tweets'].astype(int)

# Has less than 20 followers
df['less_than_20_followers'] = df['followers_count'] <= 20
df['less_than_20_followers'] = df['less_than_20_followers'].astype(int)

# Has description
df['description'] = df['description'].astype(str)
df['has_description'] = df['description'] != '0'
df['has_description'] = df['has_description'].astype(int)


# Has profile background image
df['has_profile_background_image'] = df['profile_use_background_image'] == 1.0
df['has_profile_background_image'] = df['has_profile_background_image'].astype(int)

# Has geolocation enabled
df['has_geo_enabled'] = df['geo_enabled'] == 1.0
df['has_geo_enabled'] = df['has_geo_enabled'].astype(int)


df = df[[
    '20x_friends_followers',
    '100x_friends_followers',
    '100x_followers_listed',
    'is_in_a_list',
    'lang_is_En',
    'has_never_favorited',
    '5x_followers_tweets',
    'less_than_20_followers',
    'has_description',
    'has_profile_background_image',
    'has_geo_enabled',
    'is_bot'
]]
print(df.head())


print(df.info())

df.to_csv('data/test_users.csv')
