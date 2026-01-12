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










