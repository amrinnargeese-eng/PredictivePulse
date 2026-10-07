import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Predictive Analytics", page_icon="📈", layout="wide")

st.title("📈 Predictive Analytics Using Historical Data")
st.caption("Time-series forecasting of monthly air passenger traffic")

@st.cache_data
def load_data():
    df = pd.read_csv("air_passengers.csv", parse_dates=["Date"])
    df = df.dropna(subset=["Date", "Passengers"]).sort_values("Date")
    df["Passengers"] = pd.to_numeric(df["Passengers"], errors="coerce")
    df = df.dropna()
    return df

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

df = load_data()

st.sidebar.header("Forecast Settings")
horizon = st.sidebar.slider("Future months to forecast", 3, 36, 12)
test_size = st.sidebar.slider("Test months for evaluation", 12, 36, 24)

st.subheader("1. Historical Data")
c1, c2, c3 = st.columns(3)
c1.metric("Records", len(df))
c2.metric("Start", df["Date"].min().strftime("%b %Y"))
c3.metric("End", df["Date"].max().strftime("%b %Y"))

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(df["Date"], df["Passengers"], marker="o", linewidth=1.5)
ax.set_title("Historical Monthly Passenger Traffic")
ax.set_xlabel("Date")
ax.set_ylabel("Passengers (thousands)")
ax.grid(alpha=0.25)
st.pyplot(fig, use_container_width=True)

st.subheader("2. Data Preparation")
st.write("The dataset is sorted chronologically, missing values are removed, and the monthly frequency is preserved.")
st.dataframe(df.tail(12), use_container_width=True)

# Train/test split
train = df.iloc[:-test_size].copy()
test = df.iloc[-test_size:].copy()

st.subheader("3. Model Training & Evaluation")
st.write("Model: Holt-Winters Exponential Smoothing with additive trend and 12-month seasonality.")

model = ExponentialSmoothing(
    train["Passengers"],
    trend="add",
    seasonal="add",
    seasonal_periods=12,
    initialization_method="estimated"
).fit(optimized=True)

pred = model.forecast(test_size)
mae = mean_absolute_error(test["Passengers"], pred)
rmse = np.sqrt(mean_squared_error(test["Passengers"], pred))
mape_value = mape(test["Passengers"], pred)

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:.2f}")
m2.metric("RMSE", f"{rmse:.2f}")
m3.metric("MAPE", f"{mape_value:.2f}%")

fig2, ax2 = plt.subplots(figsize=(11, 4))
ax2.plot(train["Date"], train["Passengers"], label="Training")
ax2.plot(test["Date"], test["Passengers"], label="Actual")
ax2.plot(test["Date"], pred, label="Predicted")
ax2.set_title("Actual vs Predicted — Test Period")
ax2.set_xlabel("Date")
ax2.set_ylabel("Passengers (thousands)")
ax2.legend()
ax2.grid(alpha=0.25)
st.pyplot(fig2, use_container_width=True)

# Final model on all historical data
final_model = ExponentialSmoothing(
    df["Passengers"],
    trend="add",
    seasonal="add",
    seasonal_periods=12,
    initialization_method="estimated"
).fit(optimized=True)

future_values = final_model.forecast(horizon)
future_dates = pd.date_range(
    df["Date"].max() + pd.offsets.MonthBegin(1),
    periods=horizon,
    freq="MS"
)
forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_values.values})

st.subheader("4. Future Forecast")
fig3, ax3 = plt.subplots(figsize=(11, 4))
ax3.plot(df["Date"], df["Passengers"], label="Historical")
ax3.plot(forecast_df["Date"], forecast_df["Forecast"], marker="o", label="Forecast")
ax3.axvline(df["Date"].max(), linestyle="--", linewidth=1)
ax3.set_title(f"Next {horizon} Months Forecast")
ax3.set_xlabel("Date")
ax3.set_ylabel("Passengers (thousands)")
ax3.legend()
ax3.grid(alpha=0.25)
st.pyplot(fig3, use_container_width=True)

st.dataframe(
    forecast_df.assign(
        Date=forecast_df["Date"].dt.strftime("%Y-%m"),
        Forecast=forecast_df["Forecast"].round(2)
    ),
    use_container_width=True
)

csv = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Forecast CSV",
    data=csv,
    file_name="future_forecast.csv",
    mime="text/csv"
)

st.success("Forecast generated successfully using historical monthly patterns.")
