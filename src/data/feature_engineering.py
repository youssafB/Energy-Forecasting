from utilsforecast.feature_engineering import pipeline, fourier
from functools import partial




def date_features():
    return ['hour', 'dayofweek', 'dayofyear', 'month', 'quarter','year']




def lags():
    return [1, 24]


def exg_features():
    # Returns a list of partial functions for MLForecast
    return [partial(fourier, season_length=24, k=2)]



