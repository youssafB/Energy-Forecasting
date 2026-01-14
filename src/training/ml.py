
import sys
from pathlib import Path
sys.path.append(str(Path("..").resolve()))

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from mlforecast import MLForecast
from src.data.feature_engineering import date_features, lags


# ===============================
# Model Training Script
# ===============================
# This script trains MLForecast models using
# predefined lag and date feature configurations.









def train_mlforecast_models(train_df, freq='h'):
    """
    Train MLForecast models using lag and date feature definitions.
    """
    models = {
        #'lreg': LinearRegression(),
        'dt': DecisionTreeRegressor(),
        'xgb': xgb.XGBRegressor()
    }

    ml = MLForecast(
        freq=freq,
        models=models,
        lags=lags(),
        date_features=date_features()
    )

    ml.fit(train_df, static_features=[])

    return ml

