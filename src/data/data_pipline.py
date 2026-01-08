# ===============================
# Data Pipeline Script
# ===============================
"""
Script: data_pipeline.py
Author: Youssef Bouraha
Date: 2026-01-08
Purpose: Load, preprocess, and prepare PJME hourly electricity data 
         for time series forecasting. 

Functionality:
1. Load raw CSV data and clean it using `prepare_df`.
2. Split the cleaned data into training and testing sets by date.
3. Apply exogenous and engineered features (Fourier) 
   using the pipeline function.
4. Save the processed train and test datasets to the 
   `data/preprocessed` folder for downstream modeling.

Usage:
    python -m src.data.data_pipeline  # solev the paht problem 

Notes:
- Paths are relative to project root.
- Excludes raw data from version control; only preprocessed CSVs are saved.
- Designed for reproducible, modular, and scalable ML workflows.
"""

import pandas as pd
import os 
from pathlib import Path
from data_preprocessing import prepare_df, train_test_split
from feature_engineering import exg_features
from utilsforecast.feature_engineering import pipeline  # applies features to data


# -------------------------------
# 1. Load and preprocess data
# -------------------------------


PROJECT_ROOT = Path.cwd() 
raw_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'PJME_hourly.csv')   



df = pd.read_csv(raw_path)  # raw CSV  
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
# 4. Save preprocessed data
# -------------------------------
pres_path= r'C:\Users\Guest\Desktop\ds-projects\Energy-Forecasting\data\preprocessed'

train_path = os.path.join(PROJECT_ROOT, 'data', 'preprocessed', 'train.csv')  
test_path = os.path.join(PROJECT_ROOT, 'data', 'preprocessed', 'test.csv')    

exg_train.to_csv(train_path, index=False)
exg_test.to_csv(test_path, index=False)

print('✅ Data successfully saved!')
print('✅ Train and test data are successfully loaded and ready for use.')

