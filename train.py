from pipelines.training_pipeline import run_training_pipeline
import os
from pathlib import Path
from utilsforecast.plotting import plot_series
import matplotlib.pyplot as plt



config =  {   'data_path': os.path.join( Path.cwd() , 'data', 'raw', 'PJME_hourly.csv'),
               'split_date':'2018-08-02',
               'freq':'h',
               'horizon':50,               'save_dir': os.path.join( Path.cwd() , 'trained_models')
                
} 


plot_number = 1
full_path = os.path.join('results', f"forecast_plot_{plot_number}.png" )


ml, pred_df ,  eval_df , test = run_training_pipeline(config)
ml.save(config['save_dir'])
print(f'✅ Trained models saved to {config["save_dir"]}')


# 3.2 plotting
fig= plot_series(df=test, 
                forecasts_df=pred_df, 
                palette='viridis', 
                models=['xgb','dt'])

# 4) Save the current figure to that path
fig.savefig(full_path, dpi=300, bbox_inches="tight")
print(f'✅ Forecast plot saved to {full_path}')
print(eval_df)
