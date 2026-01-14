
# ===============================
# Feature Engineering Setup
# ===============================

# This script is responsible for:
# - DEFINING which lag and date features should be used
# - CREATING Fourier (seasonal) features
#
# Lag features and date features are ONLY defined here.
# They are NOT created in this script.
# Their actual creation is handled later by MLForecast
# Fourier features are created in this script via utilsforecast.pipeline.


# It returns:
# - exg_df: historical data with created Fourier features (used for training)
# - future_df: future timestamps with Fourier features (used for forecasting)




from utilsforecast.feature_engineering import pipeline, fourier
from functools import partial





def date_features():
    return ['hour', 'dayofweek', 'dayofyear', 'month', 'quarter','year']




def lags():
    return [1, 24]


def exg_features():
    # Returns a list of partial functions for MLForecast
    return [partial(fourier, season_length=24, k=2)]




def create_features(df,freq,  horison):
    """
 
    """

    exg_df, future_df = pipeline(df, freq=freq, h=horison, features=exg_features())
    print(f'exg_df shape: {exg_df.shape}, future_df shape: {future_df.shape}')
    return exg_df, future_df










