# Air Quality Level Prediction for Pakistani Cities

## Project Overview
This project builds a machine learning model to predict air quality levels (AQI categories) using historical data from Pakistani cities. The goal is to forecast future pollution levels so authorities can issue early warnings.

## Objectives
- Predict daily AQI category (Good, Moderate, Unhealthy, etc.)
- Forecast 3 days ahead
- Handle time-series data with environmental features

## Project Structure
```
├── data/                   # Raw and processed datasets
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code
│   ├── data_processing.py # Data cleaning and preprocessing
│   ├── feature_engineering.py # Feature creation
│   ├── model_training.py  # Model training and evaluation
│   └── prediction.py      # Prediction utilities
├── models/                # Saved trained models
├── outputs/               # Prediction results
├── app.py                 # Streamlit application
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Dataset Sources
1. Pakistan Air Quality & Weather Data (2021-2024)
2. Islamabad Air Quality Data (EPA source)
3. Air Quality Index - Pakistan (various cities)
4. Air Quality Monitoring Dataset (Pakistan 2018)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Run data processing: `python src/data_processing.py`
2. Train model: `python src/model_training.py`
3. Launch Streamlit app: `streamlit run app.py`

## Model Performance
- Accuracy: TBD
- F1-Score: TBD
- RMSE: TBD
- Forecast horizon: 3 days

## Limitations
- Data quality depends on sensor reliability
- Weather patterns can change rapidly
- Limited historical data for some cities