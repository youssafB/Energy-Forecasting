from pipelines.ml_pipeline import run_ml_pipeline
from pipelines.cv_pipeline import run_cv_pipeline

import os
from pathlib import Path
from utilsforecast.plotting import plot_series
import matplotlib.pyplot as plt



config =  {   'data_path': os.path.join( Path.cwd() , 'data', 'raw', 'PJME_hourly.csv'),
               'split_date':'2018-07-20',
               'freq':'h',
               'horizon':50,               
               'save_dir': os.path.join( Path.cwd() , 'trained_models'),
               'n_windows':2
                
} 




# 1. Run the training pipeline

plot_number = 1
full_path = os.path.join('results', f"forecast_plot_{plot_number}.png" )


#ml, pred_df ,  eval_df , test = run_ml_pipeline(config)


test = run_cv_pipeline(config)

print(eval_df)
#ml.save(config['save_dir'])
#print(f'✅ Trained models saved to {config["save_dir"]}')

# 3.2 plotting
fig= plot_series(df=test,  forecasts_df=pred_df,  palette='viridis')

# 4) Save the current figure to that path
fig.savefig(full_path, dpi=300, bbox_inches="tight")
print(f'✅ Forecast plot saved to {full_path}')

