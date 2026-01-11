
import optuna

def xgboost_space(trial):
 
    return {

       # Tuned parameters
       'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
       'n_estimators': trial.suggest_int('n_estimators', 400, 1000),
        
        # Fixed parameters
        'random_state': 42,
       
    }

