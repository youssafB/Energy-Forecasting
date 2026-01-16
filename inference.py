from pipelines.inference_pipeline import run_ml_inference



import os
from pathlib import Path




config =  {   'data_path': os.path.join( Path.cwd() , 'data', 'raw', 'PJME_hourly.csv'),
               'split_date':'2018-07-20',
               'freq':'h',
               'horizon':50,               
               'save_dir': os.path.join( Path.cwd() ,'results', 'saved_models'),
               'n_windows':2,
               'plot_path' :  os.path.join('results','saved_plots', f"forecast_plot.png" ),
               'model_path': os.path.join( Path.cwd() ,'results', 'saved_models','custom_xgb'),
               
                
} 




# Forecast 24 steps ahead
pred_df = run_ml_inference(config, horizon=50)
print(pred_df.head())