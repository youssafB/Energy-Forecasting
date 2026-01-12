
import optuna



# ===============================
#  1 model spaces 
# ===============================

def xgboost_space(trial):
 
    return {

       # Tuned parameters
       'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
       'n_estimators': trial.suggest_int('n_estimators', 400, 1000),
        
        # Fixed parameters
        'random_state': 42,
       
    }




# ===============================
#  2 featurres spaces 
# ===============================

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




# ===============================
#  3 fit config
# ===============================

def fit_config(trial):
    return {
        "static_features": []
    }
