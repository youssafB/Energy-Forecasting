# ===============================
# MLForecast Training Script
# ===============================

import pandas as pd
import os
from pathlib import Path
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from utilsforecast.losses import mae
from mlforecast import MLForecast
from src.data.feature_engineering import date_features, lags
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae
from utilsforecast.plotting import plot_series
import matplotlib.pyplot as plt




print(f'Loading data...')
PROJECT_ROOT = Path.cwd() 
train_path = os.path.join(PROJECT_ROOT, 'data', 'preprocessed', 'train.csv')  
test_path = os.path.join(PROJECT_ROOT, 'data', 'preprocessed', 'test.csv')  
train = pd.read_csv(train_path, parse_dates=['ds'])
test = pd.read_csv(test_path, parse_dates=['ds'])
print(f'✅ Train shape: {train.shape}, Test shape: {test.shape}')






# ===============================
# 1 Models 
# ===============================


models = {
                'lreg': LinearRegression(),
                'dt': DecisionTreeRegressor(),
                'xgb': xgb.XGBRegressor()
}


# ===============================
# 
# ===============================

ml = MLForecast( models=models,
                 freq='h',
                 lags=[1, 24],
                 date_features=['dayofweek', 'hour'])

print(f'.......Training models using cross validation...')

cv_df = ml.cross_validation(
                h=20,
                df=train[:2000],
                n_windows=2,
                step_size=2,
                refit=True,
                static_features=[]
)
print(f'cross validation results', cv_df.head())
print(f'✅ Models trained successfully!')


# ===============================
# Compute MAE per series and per horizon
# ===============================

evaluation = evaluate(cv_df.drop(['cutoff'], axis=1), metrics=[mae])
evaluation.head()
print(f'✅  models errors', evaluation.head())




fig = plot_series(
                df=train,
                forecasts_df=cv_df.drop(['cutoff'], axis=1),
                palette='viridis',
                models=['xgb']
)

fig.savefig("results/forecast_plot.png", dpi=300)

# ===============================
# Save the trained models
# ===============================

#model_path = os.path.join(PROJECT_ROOT, 'models', 'mlforecast_models.pkl')

#ml.save(model_path)
#print(f'✅ Models saved to {model_path}')