from pipelines.ml_pipeline import run_ml_pipeline
from pipelines.cv_pipeline import run_cv_pipeline
from pipelines.fn_pipeline import run_fn_pipeline
from src.evaluation.plot import plot_and_save

import os
from pathlib import Path
from utilsforecast.plotting import plot_series
import matplotlib.pyplot as plt



config =  {   'data_path': os.path.join( Path.cwd() , 'data', 'raw', 'PJME_hourly.csv'),
               'split_date':'2018-07-20',
               'freq':'h',
               'horizon':50,               
               'save_dir': os.path.join( Path.cwd() , 'trained_models'),
               'n_windows':2,
               'plot_path' :  os.path.join('results', f"forecast_plot.png" )
                
} 







# 1  train with ml pipeline 
#pred_df ,  eval_df ,  ml  = run_ml_pipeline(config, save_models=True)

# 2  train  with cross valdiation 
cv_df ,   eval_df  = run_cv_pipeline(config)


# 3  train  with fine tuning 
cv_df ,   eval_df  = run_fn_pipeline(config)



print(eval_df.head() )


