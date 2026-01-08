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
                 lags=lags(),
                 date_features=date_features())

print(f'Training models...')
ml.fit(train, static_features=[])
print(f'✅ Models trained successfully!')



# ===============================
# Save the trained models
# ===============================

#model_path = os.path.join(PROJECT_ROOT, 'models', 'mlforecast_models.pkl')

#ml.save(model_path)
#print(f'✅ Models saved to {model_path}')