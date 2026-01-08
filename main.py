# ===============================
# MLForecast Training Script
# ===============================

import pandas as pd
import os
from pathlib import Path




PROJECT_ROOT = Path.cwd() 


train_path = os.path.join(PROJECT_ROOT, 'data', 'preprocessed', 'train.csv')  
test_path = os.path.join(PROJECT_ROOT, 'data', 'preprocessed', 'test.csv')  

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f'✅ Train and test data are successfully loaded and ready for use.')
print(f'Train shape: {train.shape}, Test shape: {test.shape}')
print("Train data sample:")
print(train.head())