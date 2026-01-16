

import sys
import os
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.loader import load_data
from src.data.data_preprocessing import prepare_df
from src.data.feature_engineering import create_features
from src.training.ml import train_ml_models




# ===============================
# MLForecast Inference Pipeline
# ===============================

import pandas as pd
from pathlib import Path
from src.data.loader import load_data
from src.data.data_preprocessing import prepare_df, train_test_split
from src.data.feature_engineering import create_features
from mlforecast import MLForecast

def run_ml_inference(config, horizon):
    """
    Inference-only pipeline for MLForecast.

    Args:
        config (dict): Configuration dictionary containing:
            - 'data_path': path to historical data CSV
            - 'model_path': path to saved MLForecast model folder
            - 'freq': data frequency string (e.g., 'H')
        horizon (int): Number of steps to forecast ahead.

    Returns:
        pred_df (pd.DataFrame): Forecasted values for the horizon.
    """

    # 1️⃣ Load historical data
    df = load_data(config["data_path"])
    df = prepare_df(df)
    print(f"Loaded historical data: {df.shape}")



    train, test = train_test_split(df, config['split_date'])
    print(f'Train shape: {train.shape}, Test shape: {test.shape}')

    h =len(test) 


    # 2️⃣ Create future features (Fourier, date, etc.)
    train_df, future_df = create_features(train, freq=config['freq'], horison=horizon)
    print(f"Future features created: {future_df.shape}")

    # 3️⃣ Load saved trained MLForecast model
    ml = MLForecast.load(config['model_path'])
    print("Loaded trained MLForecast model.")

    ml.update(train_df)

    # 4️⃣ Predict future values
    pred_df = ml.predict(h=horizon, X_df=future_df)
    print(f"Predictions generated for horizon: {horizon}")

    return pred_df, train_df





