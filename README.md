# Energy-Forecasting


# ⚡ Energy Forecasting - Time Series ML Project

A modular, scalable machine learning pipeline for time series forecasting using Nixtla's MLForecast library with automated hyperparameter tuning.

## 📋 Project Overview

This project implements an end-to-end forecasting solution for energy consumption data using:
- **Nixtla MLForecast**: Automated feature engineering and model selection
- **XGBoost**: Gradient boosting for time series prediction
- **AutoML**: Automated hyperparameter tuning
- **Modular Design**: Clean, maintainable, and scalable code structure

## 🏗️ Project Structure

```
energy-forecasting/
│
├── config/                    # Configuration files
│   ├── paths.py              # Path management
│   └── config.yaml           # Project settings
│
├── data/
│   ├── raw/                  # Original datasets
│   ├── processed/            # Cleaned data
│   └── predictions/          # Model outputs
│
├── src/                      # Source code
│   ├── data/                 # Data processing modules
│   │   ├── loader.py         # Data loading
│   │   ├── preprocessing.py  # Data cleaning
│   │   ├── feature_engineering.py
│   │
│   │
│   ├── models/               # Model implementations
│   │   └── auto_forecast.py  # AutoMLForecast wrapper
│   │
│   ├── tuning/               # Hyperparameter configs
│   │   └── hyperparameters.py
│   │
│   └── evaluation/           # Evaluation tools
│       ├── metrics.py        # Performance metrics
│       └── visualization.py  # Plotting functions
│
├── scripts/                  # Executable scripts
│   ├── train.py             # Main training pipeline
│   ├── predict.py           # Prediction script
│   └── evaluate.py          # Evaluation script
│
├── notebooks/               # Jupyter notebooks
│   └── exploratory_analysis.ipynb
│
└── outputs/                 # Generated outputs
    ├── models/              # Saved models
    ├── plots/               # Visualizations
    └── reports/             # Evaluation reports
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/energy-forecasting.git
cd energy-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Run complete training pipeline
python scripts/train.py

# Make predictions only
python scripts/predict.py --model outputs/models/your_model.pkl

# Evaluate existing predictions
python scripts/evaluate.py --predictions data/predictions/pred.csv
```

## 📊 Features

- ✅ **Automated Feature Engineering**: Date features, lags, Fourier transforms
- ✅ **Hyperparameter Optimization**: Automated tuning using Optuna
- ✅ **Cross-Validation**: Time series cross-validation for robust evaluation
- ✅ **Multiple Metrics**: MAE, MSE, RMSE support
- ✅ **Visualization**: Forecast plots and residual analysis
- ✅ **Modular Design**: Easy to extend and maintain
- ✅ **Logging**: Comprehensive logging throughout pipeline

## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Model parameters
- Feature engineering settings
- Train/test split dates
- Output paths

## 📈 Usage Example

```python
from src.data.loader import DataLoader
from src.models.auto_forecast import AutoForecastModel

# Load data
loader = DataLoader()
df = loader.load_raw_data("data/raw/PJME_hourly.csv")

# Train model
model = AutoForecastModel(models=..., freq='h', ...)
model.fit(df, n_windows=2, h=168, num_samples=10)

# Predict
predictions = model.predict(h=168)
```

## 📚 Dependencies

- pandas
- numpy
- mlforecast
- utilsforecast
- xgboost
- scikit-learn
- matplotlib
- optuna

See `requirements.txt` for complete list.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Your Name - [@yourhandle](https://twitter.com/yourhandle)

## 🙏 Acknowledgments

- [Nixtla](https://github.com/Nixtla) for MLForecast library
- Energy consumption dataset from PJM Interconnection