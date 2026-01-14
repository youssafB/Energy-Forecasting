# ===============================
# MLForecast Training Script
# ===============================
# Loads preprocessed data, defines MLForecast models, 
# performs cross-validation, and evaluates results.

import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from mlforecast import MLForecast
from src.data.feature_engineering import date_features, lags












def cross_validation(train_df,h, n_windows,  freq='h'):
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


  
    print("Training models using cross-validation...")

    cv_df = ml.cross_validation(
                h=h,
                df=train_df,    
                n_windows= n_windows,
                step_size=h,
                refit=True,
                static_features=[]
    )


    print("✅ Cross-validation complete")

    return cv_df 

