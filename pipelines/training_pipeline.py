


import sys
import os
import pandas as pd
from pathlib import Path
sys.path.append(str(Path("..").resolve()))
from src.data.loader import load_data
from src.data.data_preprocessing import prepare_df , train_test_split
from src.data.feature_engineering import create_features
from src.training.train_mlforecast import train_mlforecast_models
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae
from pathlib import Path


config =  {   'data_path': os.path.join( Path.cwd() , 'data', 'raw', 'PJME_hourly.csv'),
               'split_date':'2018-08-02',
               'freq':'h',
               'horizon':50,
               'save_dir': os.path.join( Path.cwd() , 'trained_models')
                  

   

           
           
           } 





def run_training_pipeline(config):
    # 1️⃣ Load & prepare data
    df = load_data(config["data_path"])
    df = prepare_df(df)

    train, test = train_test_split(df, config['split_date'])
    print(f'Train shape: {train.shape}, Test shape: {test.shape}')

    # 2️⃣ Feature engineering
    exg_df, future_df = create_features(train, config['freq'], config['horizon'])
    

    # 3️⃣ Train
    ml = train_mlforecast_models(exg_df, config['freq'])

    # 4️⃣ Predict
    pred_df = ml.predict(h=config['horizon'], X_df=future_df)

    # 5️⃣ Evaluate
    #evaluate_predictions(test, preds, metric='mae')

    merg_df = pd.merge(pred_df ,test, on =['unique_id', 'ds'],how='left')
    eval_df = evaluate(merg_df,metrics= [mae])
    

 


    return ml, pred_df ,  eval_df , test
    #return eval_df









#eval_df  = run_training_pipeline(config)
#print(f'Predictions shape: {eval_df.shape}')
#print(eval_df)



