import pandas as pd

# ===============================
# Data Processing Script
# ===============================
# This script contains two functions:
# 1. prepare_df: prepares raw data for MLForecast format
# 2. train_test_split: splits data into training and test sets


def prepare_df(df, unique_id='A'):
    df = df.copy()
    df['ds'] = pd.to_datetime(df['Datetime'])
    df['y'] = df['PJME_MW']
    df['unique_id'] = unique_id
    df = df[['ds', 'y', 'unique_id']].sort_values('ds')
    return df


def train_test_split(df, split_date):
    train = df[df['ds'] < split_date]
    test = df[df['ds'] >= split_date]
    return train, test
