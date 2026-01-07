import pandas as pd 



def preprocess_df(df, unique_id='A'):
    """
    Preprocess a DataFrame for MLForecast.
    
    Args:
        df (pd.DataFrame): Input DataFrame with columns ['Datetime', 'PJME_MW', ...].
        unique_id (str): The unique ID to assign for MLForecast.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame with columns ['ds', 'y', 'unique_id'].
    """
    df = df.copy()  # avoid modifying original df
    df['ds'] = pd.to_datetime(df['Datetime'])  # convert datetime
    df['y'] = df['PJME_MW']                    # rename target column
    df['unique_id'] = unique_id                # add unique_id
    df = df[['ds', 'y', 'unique_id']].sort_values('ds')  # keep only needed columns and sort
    return df
