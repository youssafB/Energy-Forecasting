
import optuna




def feature_space(trial):
    base_lags = [1, 24]
    date_features = ['hour', 'dayofweek', 'dayofyear', 'month', 'quarter','year'] 

    # Pick one extra lag to test (0 = no extra)
    extra_lag = trial.suggest_categorical("extra_lag", [0, 192, 500, 1000])

    # Final lags list
    lags = base_lags + ([extra_lag] if extra_lag != 0 else [])

    return {
        "lags": lags,
        'date_features':date_features,  # Fixed: always use these
        
       
    }



def fit_config(trial):
    return {
        "static_features": []
    }
