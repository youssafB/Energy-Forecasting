


import sys
import os
import pandas as pd
from pathlib import Path
sys.path.append(str(Path("..").resolve()))
from src.data.loader import load_data
from src.data.data_preprocessing import prepare_df , train_test_split
from src.data.feature_engineering import create_features
from src.training.cv import  cross_validation
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae








def run_cv_pipeline(config):
    
    # -------------------------------
    # 1️⃣ Load & prepare data
    # -------------------------------
    print("Loading and preparing data...")

    df = load_data(config["data_path"])
    df = prepare_df(df)

    train, test = train_test_split(df, config['split_date'])
    print(f'Train shape: {train.shape}, Test shape: {test.shape}')
    h= len(test)

    # 2️⃣ Feature engineering
    exg_df, future_df = create_features(train, config['freq'], h)
    

    # 3️⃣ Train with complex cross validation

    cv_df = cross_validation( exg_df, h, config['n_windows'])
    

    # ===============================
    # 4️⃣ Compute MAE per series and per horizon
    # ===============================

    evaluation = evaluate(cv_df.drop(['cutoff'], axis=1), metrics=[mae])
    evaluation.head()
    print(f'✅  models errors', evaluation.head())



    return  cv_df ,    evaluation