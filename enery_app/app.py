import sys
from pathlib import Path
import os
import streamlit as st
import pandas as pd

# Add parent folder to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pipelines.inference_pipeline import run_ml_inference
from src.evaluation.plot import plot_and_save

# ==========================
# Configuration
# ==========================
config = {
    'data_path': os.path.join(Path.cwd(), 'data', 'raw', 'PJME_hourly.csv'),
    'split_date': '2018-07-20',
    'freq': 'H',  # use 'H' for hourly
    'horizon': 50,
    'save_dir': os.path.join(Path.cwd(), 'results', 'saved_models'),
    'n_windows': 2,
    'plot_path': os.path.join('results', 'saved_plots', "forecast_plot.png"),
    'model_path': os.path.join(Path.cwd(), 'results', 'saved_models', 'custom_xgb')
}

# ==========================
# Streamlit UI
# ==========================
st.title("⚡ Energy Forecasting")

# Sidebar controls
st.sidebar.title("⚙️ Controls")

# Forecast horizon input
horizon = st.sidebar.slider(
    "Forecast Horizon (hours)",
    min_value=1,
    max_value=168,
    value=50,
    step=1
)

# Confidence interval input (unused here, but can be added later)
confidence = st.sidebar.number_input(
    "Confidence Interval (%)",
    min_value=50,
    max_value=99,
    value=90,
    step=1
)

# Model selection (unused, placeholder)
model_options = ["Model A", "Model B", "Model C"]
model_selected = st.sidebar.selectbox("Select Model", model_options)

# ==========================
# Run inference when button clicked
# ==========================
if st.sidebar.button("Run"):

    # --- Run MLForecast inference ---
    pred_df , df = run_ml_inference(config, horizon=horizon)  # use slider value

    # Debug: print first few rows
    print(pred_df.head())

    # --- Ensure datetime is correct ---
    pred_df['ds'] = pd.to_datetime(pred_df['ds'])
    

    # --- Set index to datetime for line chart ---
    pred_df = pred_df.set_index('ds')


    # --- Plot predictions using Streamlit line_chart ---
    st.line_chart(pred_df['custom_xgb'])


    

    fig = plot_and_save(df, pred_df )
    # display it in Streamlit
    st.pyplot(fig)
    

