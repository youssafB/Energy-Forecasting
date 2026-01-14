
import sys
from pathlib import Path
sys.path.append(str(Path("..").resolve()))
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.tuning .tuning_spaces import feature_space, fit_config, xgboost_space
from mlforecast.auto import AutoMLForecast
from mlforecast.auto import AutoModel








def fine_tune(train_df,h, freq='h', n_windows=2):
    """
    Train MLForecast and adding lags and date features.
    """
    # Initialize models
    models = {'custom_xgb': AutoModel(model=xgb.XGBRegressor(), config=xgboost_space)}


    # # Configure AutoMLForecast with both model and feature tuning
    auto_mlf = AutoMLForecast(
                freq=freq,
                models=models,
                init_config=feature_space,    
                fit_config=fit_config
    )

    # Train
    
    auto_mlf.fit(df=train_df, 
                     n_windows= 2,
                     h= h, 
                     num_samples= 10,
                     
 )

    # Print names of trained models
    trained_model_names = ', '.join(models.keys())
    print(f'✅ Model(s) trained successfully: {trained_model_names}')

    return auto_mlf





