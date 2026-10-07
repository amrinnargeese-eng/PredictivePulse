# Predictive Analytics Using Historical Data

## Internship Project

This project builds a time-series predictive analytics system that uses historical monthly air passenger data to forecast future passenger traffic.

### Objective
- Clean and preprocess historical data
- Analyze historical trends and seasonality
- Train a time-series forecasting model
- Evaluate prediction accuracy
- Visualize actual vs predicted values
- Forecast future monthly values

### Dataset
`air_passengers.csv` contains monthly passenger counts from January 1949 to December 1960.

### Model
**Holt-Winters Exponential Smoothing**
- Additive trend
- Additive seasonality
- 12-month seasonal period

The project first holds out the final 24 months for evaluation. It calculates:
- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- MAPE — Mean Absolute Percentage Error

After evaluation, the final model is retrained on the complete historical dataset and used to forecast the requested number of future months.

## Project Structure

```text
predictive_analytics_historical_data/
├── app.py
├── air_passengers.csv
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
└── .gitignore
```

## Installation

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Linux/macOS:
```bash
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run:
```bash
streamlit run app.py
```

## GitHub Repository Name

`predictive-analytics-historical-data`

## Suggested Resume Description

> Developed a time-series predictive analytics dashboard using Python, Streamlit, Pandas, Matplotlib, Scikit-learn and Holt-Winters Exponential Smoothing to analyze historical passenger traffic, evaluate forecasting accuracy using MAE/RMSE/MAPE, visualize trends and generate future forecasts.

## Viva Questions

1. What is predictive analytics?
2. Why is historical data important for forecasting?
3. What is time-series data?
4. What are trend and seasonality?
5. Why was Holt-Winters selected?
6. What is train-test splitting in time-series forecasting?
7. What is MAE?
8. What is RMSE?
9. What is MAPE?
10. Why should future forecasting be evaluated on unseen data?
