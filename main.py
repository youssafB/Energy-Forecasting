from src.data_preprocessing import  preprocess_df
import pandas as pd 






df = pd.read_csv('data/PJME_hourly.csv')
data = preprocess_df(df)
print(data.head())









