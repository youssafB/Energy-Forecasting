

import pandas as pd

# ===============================
# Data Loading Script
# ===============================
# This script is responsible for loading data.




def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)
