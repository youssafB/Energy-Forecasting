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
2. Apply exogenous and engineered features (Fourier) 
   using the pipeline function.
4. Save the processed datasets to the 
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
from data_preprocessing import prepare_df
from feature_engineering import exg_features
from utilsforecast.feature_engineering import pipeline  # applies features to data


# -------------------------------
# 1. Load and preprocess data
# -------------------------------


PROJECT_ROOT = Path.cwd() 
raw_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'PJME_hourly.csv')   



df= pd.read_csv(raw_path)  # raw CSV  
df = prepare_df(df)                      # clean, rename, add unique_id
print("Preprocessed data sample:")
print(df.head())


# -------------------------------
# 2. Apply exogenous features
# -------------------------------
# exg_features() returns list of feature functions (Fourier, etc.)
exg_df, future_df = pipeline(df, freq='h', features=exg_features())
print(f'Exogenous df shape: {exg_df.shape}')




# -------------------------------
# 3. Save preprocessed data
# -------------------------------

pre_path = os.path.join(PROJECT_ROOT, 'data', 'preprocessed', 'data.csv')     
exg_df.to_csv(pre_path, index=False)


print('✅ Data successfully saved!')


