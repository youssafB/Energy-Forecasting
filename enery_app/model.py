import pandas as pd

def predict():
    HORIZON = 24

    # Get history + future from predict function
    csv_path= r"C:\Users\Guest\Desktop\ds-projects\Energy-Forecasting\enery_app\data\synthetic_energy_data.csv"

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # split dataframe
    history = df.iloc[:-HORIZON]
    future = df.iloc[-HORIZON:]

    return history, future




import matplotlib.pyplot as plt

def plot_history_future(history, future, title="Energy Load Forecast"):
    """
    Plot history and future in one chart.
    
    Parameters:
    - history: pd.DataFrame with columns ["timestamp", "load_mw"]
    - future: pd.DataFrame with columns ["timestamp", "load_mw"]
    - title: str, chart title
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # History: solid line
    ax.plot(history["timestamp"], history["load_mw"], label="History", color="blue")

    # Future: dashed line
    ax.plot(future["timestamp"], future["load_mw"], label="Future", color="orange", linestyle="--")

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Load (MW)")
    ax.set_title(title)
    ax.legend()

    return fig


import pandas as pd
import streamlit as st

import pandas as pd
import streamlit as st


import pandas as pd

def prepare_forecast_plot_df(
    hist_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    time_col: str = "ds",
    target_col: str = "y",
    pred_col: str = "custom_xgb",
    history_window: int | None = None,
):
    """
    Combine historical data and forecast data into one DataFrame
    for Streamlit line_chart.

    Parameters
    ----------
    hist_df : pd.DataFrame
        Historical data with time and target columns
    pred_df : pd.DataFrame
        Forecast data with time and prediction column
    time_col : str
        Time column name (default: 'ds')
    target_col : str
        Historical target column name (default: 'y')
    pred_col : str
        Prediction column name (default: 'custom_xgb')
    history_window : int | None
        If provided, only the last N historical points are used

    Returns
    -------
    pd.DataFrame
        Indexed by time_col with columns ['history', 'forecast']
    """

    # Optional: limit history length
    if history_window is not None:
        hist_df = hist_df.tail(history_window)

    # Prepare data
    hist_plot = hist_df[[time_col, target_col]].rename(
        columns={target_col: "history"}
    )
    pred_plot = pred_df[[time_col, pred_col]].rename(
        columns={pred_col: "forecast"}
    )

    # Combine and sort
    plot_df = pd.concat([hist_plot, pred_plot])
    plot_df = plot_df.set_index(time_col).sort_index()

    return plot_df

