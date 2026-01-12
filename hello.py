import pandas as pd
import sys
import os
from pathlib import Path
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae
sys.path.append(str(Path("..").resolve()))
from src.data.feature_engineering import date_features, lags, partial
from src.data.data_preprocessing import prepare_df
from utilsforecast.feature_engineering import pipeline, fourier
from functools import partial
from src.data.feature_engineering import  exg_features
from fine_tuning.models import xgboost_space
from fine_tuning.features import feature_space, fit_config
from mlforecast.auto import AutoMLForecast
from mlforecast.auto import AutoModel




# -------------------------------
# 1.1 Data Prepartion for Nixtla
# -------------------------------


raw_path =r'C:\Users\Guest\Desktop\ds-projects\Energy-Forecasting\data\raw\PJME_hourly.csv'


df= pd.read_csv(raw_path)  # raw CSV  
df = prepare_df(df)                      # clean, rename, add unique_id
df.head()


# -------------------------------
# 1.2 Split into train and test
# -------------------------------
split_date = '2018-08-02'

train = df[df['ds'] < split_date ]
test  = df[df['ds'] >= split_date ]

print(f'Train shape: {train.shape}, Test shape: {test.shape}')




# -------------------------------
# 2. Apply exogenous features
# -------------------------------
# exg_features() returns list of feature functions (Fourier, etc.)




exg_df, future_df = pipeline(train , freq='h', h=len(test), features=exg_features())
print(f'Exogenous df shape: {exg_df.shape}')




# ===============================
# 
# ===============================


# Configure AutoMLForecast with both model and feature tuning


auto_mlf = AutoMLForecast(
                models={'custom_xgb': AutoModel(model=xgb.XGBRegressor(), config=xgboost_space)}, 
                freq='h',                                             
                init_config=feature_space,    
                fit_config=fit_config
                
                
                
            )




auto_mlf.fit(df=exg_df , 
                     n_windows= 2,
                     h= len(test), 
                     num_samples= 10,
                     
 )
#print(f'cross validation results', cv_df.head())
#print(f'✅ Models trained successfully!')



df_pred = auto_mlf.predict(h=len(future_df), 
                           X_df=future_df,
                           )
df_pred.head()



from utilsforecast.evaluation import evaluate


merg_df = pd.merge(df_pred,test, on =['unique_id', 'ds'],how='left')
eval_df = evaluate(merg_df,metrics= [mae])
eval_df.head()




# 3.2 plotting

from utilsforecast.plotting import plot_series


plot_series(df=test, 
                forecasts_df=df_pred[-200:], 
                max_insample_length=200, 
                palette='viridis', 
              )