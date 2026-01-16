
from utilsforecast.plotting import plot_series
import matplotlib.pyplot as plt




def plot_and_save(
    df, 
    forecasts_df=None, 
    #full_path, 
    palette='viridis', 
    models=None
):
    """
    Plot forecasts and save the figure.

    Parameters:
        df (pd.DataFrame): The main dataframe to plot.
        forecasts_df (pd.DataFrame): The forecast dataframe.
        full_path (str): Path to save the figure.
        palette (str): Color palette for plotting.
        models (list, optional): List of models to include in the plot.
    """
    fig = plot_series(df=df, forecasts_df=forecasts_df, palette=palette, models=models)
    #fig.savefig(full_path, dpi=300, bbox_inches="tight")
    #print(f'✅ Forecast plot saved to {full_path}')
    return fig
