

# ===============================
#  ml_pipeline Script
# ===============================
# Loads data, prepares it, creates features, trains MLForecast models,
# makes predictions, evaluates them, and returns:
# - ml: trained MLForecast model object
# - pred_df: predicted values for the forecast horizon
# - eval_df: evaluation results (e.g., MAE)
# - test: test set used for evaluation






import sys
import os
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.loader import load_data
from src.data.data_preprocessing import prepare_df, train_test_split
from src.data.feature_engineering import create_features
from src.training.ml import train_mlforecast_models
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae











def run_ml_pipeline(config):
    # 1️⃣ Load & prepare data
    df = load_data(config["data_path"])
    df = prepare_df(df)
    train, test = train_test_split(df, config['split_date'])
    print(f'Train shape: {train.shape}, Test shape: {test.shape}')

    h =len(test) 

    # 2️⃣ Feature engineering
    exg_df, future_df = create_features(train, config['freq'],  h)

    # 3️⃣ Train models
    ml = train_mlforecast_models(exg_df, config['freq'])

    # 4️⃣ Predict
    pred_df = ml.predict(h=h , X_df=future_df)

    # 5️⃣ Evaluate
    merged_df = pd.merge(pred_df, test, on=['unique_id', 'ds'], how='left')
    eval_df = evaluate(merged_df, metrics=[mae])

    return ml, pred_df, eval_df, test



"""
#Example usage (uncomment to run)

# Config dictionary
config = {
    'data_path': os.path.join(Path.cwd(), 'data', 'raw', 'PJME_hourly.csv'),
    'split_date': '2018-07-20',
    'freq': 'h',
    'save_dir': os.path.join(Path.cwd(), 'trained_models')
}


ml_model, predictions, evaluation, test_data = run_ml_pipeline(config)
print(f'Predictions shape: {predictions.shape}')
print(evaluation)
"""