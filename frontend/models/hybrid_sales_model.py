import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import xgboost as xgb
import streamlit as st

def hybrid_sales_forecast_plot(df, sales_col, date_col, forecast_periods=30):
    """
    Hybrid Prophet + XGBoost forecasting with plotting inside the function.
    No need to unpack, just call it in Streamlit.

    Parameters
    ----------
    df : pd.DataFrame
        Sales data
    sales_col : str
        Column name for sales
    date_col : str
        Column name for date
    forecast_periods : int
        Number of days to forecast
    """
    df_plot = df[[date_col, sales_col]].copy()
    df_plot[date_col] = pd.to_datetime(df_plot[date_col])
    df_plot = df_plot.rename(columns={date_col: "ds", sales_col: "y"})

    # Prophet
    prophet_model = Prophet()
    prophet_model.fit(df_plot)
    future = prophet_model.make_future_dataframe(periods=forecast_periods, freq='D')
    forecast = prophet_model.predict(future)

    # Residuals for XGBoost
    df_plot = df_plot.merge(forecast[['ds','yhat']], on='ds', how='left')
    df_plot['residual'] = df_plot['y'] - df_plot['yhat']
    df_plot['lag1'] = df_plot['y'].shift(1)
    df_plot['lag7'] = df_plot['y'].shift(7)
    df_plot['dayofweek'] = df_plot['ds'].dt.dayofweek
    df_plot['month'] = df_plot['ds'].dt.month
    df_plot = df_plot.dropna()

    x = df_plot[['lag1','lag7','dayofweek','month']]
    y = df_plot['residual']
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    xgb_model.fit(x, y)

    # Future features
    future_features = pd.DataFrame({
        'lag1': np.repeat(df_plot['y'].iloc[-1], forecast_periods),
        'lag7': np.repeat(df_plot['y'].iloc[-7], forecast_periods),
        'dayofweek': future['ds'].iloc[-forecast_periods:].dt.dayofweek.values,
        'month': future['ds'].iloc[-forecast_periods:].dt.month.values
    })
    future_residuals = xgb_model.predict(future_features)

    # Hybrid forecast
    forecast_hybrid = forecast.tail(forecast_periods).copy()
    forecast_hybrid['hybrid_forecast'] = forecast_hybrid['yhat'] + future_residuals

    # Plot
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df_plot['ds'], df_plot['y'], label='Actual Sales', color='black')
    ax.plot(forecast_hybrid['ds'], forecast_hybrid['yhat'], label='Prophet Forecast', color='blue', linestyle='--')
    ax.plot(forecast_hybrid['ds'], forecast_hybrid['hybrid_forecast'], label='Hybrid Forecast', color='red')

    if 'yhat_lower' in forecast_hybrid.columns and 'yhat_upper' in forecast_hybrid.columns:
        ax.fill_between(forecast_hybrid['ds'], forecast_hybrid['yhat_lower'], forecast_hybrid['yhat_upper'], color='blue', alpha=0.2, label='Prophet CI')

    ax.set_title("Hybrid Sales Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)

    # Display table
    st.write("### Forecast Data")
    st.dataframe(forecast_hybrid.tail(10))
