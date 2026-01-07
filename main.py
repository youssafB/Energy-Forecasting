# ===============================
# MLForecast Training Script
# ===============================

import pandas as pd
from src.data_preprocessing import prepare_df, train_test_split
from src.feature_engineering import date_features, lags, exg_features
from utilsforecast.feature_engineering import pipeline  # applies features to data


# -------------------------------
# 1. Load and preprocess data
# -------------------------------
df = pd.read_csv('data/PJME_hourly.csv')  # raw CSV
data = prepare_df(df)                      # clean, rename, add unique_id
print("Preprocessed data sample:")
print(data.head())

# -------------------------------
# 2. Split into train and test
# -------------------------------
split_date = '2017-01-01'
train, test = train_test_split(data, split_date)
print(f'Train shape: {train.shape}, Test shape: {test.shape}')

# -------------------------------
# 3. Apply exogenous features
# -------------------------------
# exg_features() returns list of feature functions (Fourier, etc.)
exg_train, exg_test = pipeline(train, h=len(test), freq='h', features=exg_features())
print(f'Exogenous train shape: {exg_train.shape}, Exogenous test shape: {exg_test.shape}')

# -------------------------------
# Now exg_train and exg_test can be used in MLForecast
# -------------------------------
