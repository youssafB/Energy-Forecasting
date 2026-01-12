


import sys
import os
import pandas as pd
from pathlib import Path
sys.path.append(str(Path("..").resolve()))
from src.data.loader import load_data
from src.data.data_preprocessing import prepare_df , train_test_split
from src.data.feature_engineering import create_features
from src.training.cross_validation import  train_with_cross_valid
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae



config =  {   'data_path': os.path.join( Path.cwd() , 'data', 'raw', 'PJME_hourly.csv'),
               'split_date':'2018-08-02',
               'freq':'h',
               'horizon':50,
               'n_windows' : 2

           
           
           }




def run_cv_pipeline(config):
    # 1️⃣ Load & prepare data
    df = load_data(config["data_path"])
    df = prepare_df(df)

    train, test = train_test_split(df, config['split_date'])
    print(f'Train shape: {train.shape}, Test shape: {test.shape}')

    # 2️⃣ Feature engineering
    exg_df, future_df = create_features(train, config['freq'], config['horizon'])
    

    # 3️⃣ Train with complex cross validation

    cv_df = train_with_cross_valid( exg_df, config['n_windows'], config['horizon'])


    # 4️⃣  Evaluate
    metrics = evaluate(cv_df.drop(columns=['cutoff']),  metrics=[mae], )


    return  cv_df ,   metrics