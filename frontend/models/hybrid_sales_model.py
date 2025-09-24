import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import xgboost as xgb
import streamlit as st

# Inject custom CSS styling
st.markdown("""
    <style>
    /* Background */
    .stApp {
            background: transparent !important;
        }
        video.bg-video {
            position: fixed;     /* allow scrolling */
            top: 0;
            left: 0;
            display:flex;
            align-items: center;
            justify-content: center;    
            width: 110%;           /* fill full width */
            height: 100%;           /* scale height proportionally */
            min-height: 100%;       /* ensures it covers vertically */
            object-fit: cover;    /* show entire video (no cropping) */
            z-index: -1;            /* push behind content */
            background: black;  
            background-position:center;
            background-size: cover;
        }

    /* Titles */
    h1, h2, h3 {
        color: #1a237e;
        font-weight: 600;
    }

    /* Forecast Dataframe */
    .stDataFrame {
        border: 2px solid #1a73e8;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Plot area wrapper */
    .plot-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    /* Buttons */
    .stButton > button {
        background: #1a73e8;
        color: white;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: #0b5ed7;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)


def hybrid_sales_forecast_plot(df, sales_col, date_col, forecast_periods=30):
    """
    Hybrid Prophet + XGBoost forecasting with plotting inside the function.
    """

    # Prepare data
    df_plot = df[[date_col, sales_col]].copy()
    df_plot[date_col] = pd.to_datetime(df_plot[date_col])
    df_plot = df_plot.rename(columns={date_col: "ds", sales_col: "y"})

    # Prophet model
    prophet_model = Prophet()
    prophet_model.fit(df_plot)
    future = prophet_model.make_future_dataframe(periods=forecast_periods, freq='D')
    forecast = prophet_model.predict(future)

    # Residuals for XGBoost
    df_plot = df_plot.merge(forecast[['ds', 'yhat']], on='ds', how='left')
    df_plot['residual'] = df_plot['y'] - df_plot['yhat']
    df_plot['lag1'] = df_plot['y'].shift(1)
    df_plot['lag7'] = df_plot['y'].shift(7)
    df_plot['dayofweek'] = df_plot['ds'].dt.dayofweek
    df_plot['month'] = df_plot['ds'].dt.month
    df_plot = df_plot.dropna()

    x = df_plot[['lag1', 'lag7', 'dayofweek', 'month']]
    y = df_plot['residual']
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
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
    with st.container():
        st.markdown('<div class="plot-card">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_plot['ds'], df_plot['y'], label='Actual Sales', color='black')
        ax.plot(forecast_hybrid['ds'], forecast_hybrid['yhat'], label='Prophet Forecast', color='blue', linestyle='--')
        ax.plot(forecast_hybrid['ds'], forecast_hybrid['hybrid_forecast'], label='Hybrid Forecast', color='red')

        if 'yhat_lower' in forecast_hybrid.columns and 'yhat_upper' in forecast_hybrid.columns:
            ax.fill_between(forecast_hybrid['ds'], forecast_hybrid['yhat_lower'], forecast_hybrid['yhat_upper'],
                            color='blue', alpha=0.2, label='Prophet CI')

        ax.set_title("📈 Hybrid Sales Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # Display table with style
    st.markdown("### 🔮 Forecast Data")
    st.dataframe(forecast_hybrid.tail(10).style.set_properties(
        **{'background-color': '#f1f8ff', 'color': 'black', 'border-color': 'blue'}
    ))
