
import sys
from pathlib import Path
sys.path.append(str(Path("..").resolve()))
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from mlforecast import MLForecast
from src.data.feature_engineering import date_features, lags









def train_mlforecast_models(train_df, freq='h'):
    """
    Train MLForecast and adding lags and date features.
    """
    # Initialize models
    models = {
        'lreg': LinearRegression(),
        'dt': DecisionTreeRegressor(),
        'xgb': xgb.XGBRegressor()
    }

    # Call feature functions
    lags_list = lags()
    date_feat_list = date_features()

    # Initialize MLForecast
    ml = MLForecast(
        freq=freq,
        models=models,
        lags=lags_list,
        date_features=date_feat_list
    )

    # Train
    ml.fit(train_df, static_features=[])

    # ✅ Print names of trained models
    trained_model_names = ', '.join(models.keys())
    print(f'✅ Model(s) trained successfully: {trained_model_names}')

    return ml


