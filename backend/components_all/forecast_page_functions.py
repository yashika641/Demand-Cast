import plotly.graph_objs as go
import numpy as np
import pandas as pd

def forecast_metrics(y_true, y_pred, horizon_7=True, horizon_30=True):
    """
    Compute RMSE, MAPE, and horizon-based MAPEs for 7 & 30 day windows.

    Parameters:
    -----------
    y_true : array-like
        Ground truth values
    y_pred : array-like
        Predicted values
    horizon_7 : bool
        Compute 7-day MAPE
    horizon_30 : bool
        Compute 30-day MAPE

    Returns:
    --------
    dict : {
        "rmse": float,
        "mape": float,
        "mape_7d": float,
        "mape_30d": float
    }
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ---- Core Metrics ----
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Avoid division by zero in MAPE
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

    results = {
        "rmse": float(rmse),
        "mape": float(mape)
    }

    # ---- Horizon MAPEs (for forecasting problems) ----
    if horizon_7:
        y_true_7 = y_true[:7]
        y_pred_7 = y_pred[:7]
        mape_7 = np.mean(np.abs((y_true_7 - y_pred_7) / np.where(y_true_7 == 0, 1, y_true_7))) * 100
        results["mape_7d"] = float(mape_7)

    if horizon_30 and len(y_true) >= 30:
        y_true_30 = y_true[:30]
        y_pred_30 = y_pred[:30]
        mape_30 = np.mean(np.abs((y_true_30 - y_pred_30) / np.where(y_true_30 == 0, 1, y_true_30))) * 100
        results["mape_30d"] = float(mape_30)

    return results

def get_plotly_model_comparison_dict(y_true, predictions, title="Demand Forecast Comparison"):
    """
    Creates an interactive Plotly figure dict for Actual vs Multiple Model Predictions.

    Parameters:
        y_true (array-like): Actual demand values
        predictions (dict): { "ModelName": [pred_values], ... }
        title (str): Chart title

    Returns:
        dict: Plotly figure dictionary (JSON-serializable)
    """

    # Convert to numpy
    y_true = np.array(y_true)

    # ---- Create traces ----
    traces = []

    # Actual demand line
    traces.append(
        go.Scatter(
            x=list(range(len(y_true))),
            y=y_true,
            mode="lines",
            name="Actual",
            line=dict(width=3, color="black")
        )
    )

    # Model predictions
    for model_name, y_pred in predictions.items():
        traces.append(
            go.Scatter(
                x=list(range(len(y_pred))),
                y=list(y_pred),
                mode="lines",
                name=model_name,
                line=dict(width=2, dash="dash")
            )
        )

    # ---- Layout ----
    layout = go.Layout(
        title=title,
        xaxis=dict(title="Time"),
        yaxis=dict(title="Demand"),
        hovermode="x unified",
        template="plotly_white"
    )

    # ---- Return full plot dict ----
    fig_dict = {"data": traces, "layout": layout}

    return fig_dict


def get_plotly_shap_barh_dict(shap_values, title="SHAP Feature Importance"):
    """
    Creates an interactive Plotly horizontal bar chart for SHAP feature importance.

    Parameters:
    -----------
    shap_values : dict or pandas.Series
        Feature importance values, e.g.,
        {
            "price": 0.31,
            "stock_level": 0.22,
            "promo_flag": 0.15
        }
    title : str
        Title of the chart

    Returns:
    --------
    dict : Plotly figure dictionary
    """

    # Convert to Series for easy manipulation
    if isinstance(shap_values, dict):
        shap_series = pd.Series(shap_values)
    elif isinstance(shap_values, pd.Series):
        shap_series = shap_values.copy()
    else:
        raise ValueError("shap_values must be a dict or pandas.Series")

    # Sort values descending
    shap_series = shap_series.sort_values(ascending=True)  # ascending for horizontal bar

    # ---- Create bar chart ----
    trace = go.Bar(
        x=shap_series.values,
        y=shap_series.index,
        orientation='h',
        marker=dict(color='rgba(31, 119, 180, 0.7)'),
        name="SHAP Importance"
    )

    # ---- Layout ----
    layout = go.Layout(
        title=title,
        xaxis=dict(title="Mean |SHAP| Value"),
        yaxis=dict(title="Feature"),
        template="plotly_white",
        margin=dict(l=150, r=50, t=60, b=40)
    )

    # ---- Return dict ----
    return {
        "data": [trace],
        "layout": layout
    }

def horizon_accuracy_table(y_true, y_pred):
    """
    Generates horizon-wise MAPE, RMSE, Coverage, and Status table data.
    y_true, y_pred must be arrays (numpy, list, pandas)
    
    Returns:
        list of dicts → JSON serializable table
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    horizons = [1, 7, 14, 30]
    table = []

    def calc_mape(y_t, y_p):
        return float(np.mean(np.abs((y_t - y_p) / np.where(y_t == 0, 1, y_t))) * 100)

    def calc_rmse(y_t, y_p):
        return float(np.sqrt(np.mean((y_t - y_p) ** 2)))

    def calc_coverage(y_t, y_p, tolerance=0.20):
        """
        Coverage = % of predictions whose percentage error < tolerance
        Example: tolerance=0.20 => 20% error margin
        """
        pct_error = np.abs((y_t - y_p) / np.where(y_t == 0, 1, y_t))
        covered = (pct_error <= tolerance).mean()
        return float(covered * 100)

    def status_from_mape(mape):
        if mape <= 3:
            return "Excellent"
        elif mape <= 5:
            return "Good"
        elif mape <= 8:
            return "Acceptable"
        else:
            return "Poor"

    for h in horizons:
        if len(y_true) < h:
            continue

        y_t = y_true[:h]
        y_p = y_pred[:h]

        mape = calc_mape(y_t, y_p)
        rmse = calc_rmse(y_t, y_p)
        coverage = calc_coverage(y_t, y_p)
        status = status_from_mape(mape)

        table.append({
            "horizon": f"{h} Day" if h == 1 else f"{h} Days",
            "mape": round(mape, 2),
            "rmse": round(rmse, 2),
            "coverage": round(coverage, 2),
            "status": status
        })

    return table

def zero_inflation_rate(y_true):
    y_true = np.array(y_true)
    zeros = np.sum(y_true == 0)
    return float((zeros / len(y_true)) * 100)

def croston_forecast(y, alpha=0.1):
    """
    Classic Croston forecasting for intermittent demand.
    Returns the Croston forecast for each period.
    """
    y = np.array(y)
    n = len(y)

    # Initialize
    demand = None
    interval = None
    forecast = np.zeros(n)
    last_non_zero = 0

    for t in range(n):
        if y[t] > 0:
            if demand is None:
                demand = y[t]
                interval = t + 1
            else:
                demand = alpha * y[t] + (1 - alpha) * demand
                interval = alpha * (t - last_non_zero) + (1 - alpha) * interval
            
            last_non_zero = t
        
        if demand is not None and interval is not None:
            forecast[t] = demand / interval

    return forecast


def croston_mape(y_true):
    forecast = croston_forecast(y_true)
    y_true = np.array(y_true)

    nonzero_idx = y_true > 0
    if nonzero_idx.sum() == 0:
        return None  # no nonzero values → not applicable

    mape = np.mean(np.abs((y_true[nonzero_idx] - forecast[nonzero_idx]) / y_true[nonzero_idx])) * 100
    return float(mape)

def non_zero_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    idx = y_true > 0
    if idx.sum() == 0:
        return None

    pct_error = np.abs((y_true[idx] - y_pred[idx]) / y_true[idx])
    accuracy = (1 - pct_error).mean() * 100

    return float(accuracy)

def intermittent_demand_metrics(y_true, y_pred):
    return {
        "zero_inflation_rate": zero_inflation_rate(y_true),
        "croston_mape": croston_mape(y_true),
        "non_zero_accuracy": non_zero_accuracy(y_true, y_pred)
    }

import plotly.graph_objs as go
import numpy as np

def get_intermittent_demand_plot(y_true, y_pred, title="Intermittent Demand Pattern"):
    """
    Creates an interactive Plotly graph for intermittent demand pattern.
    Returns a JSON-serializable Plotly dict usable on frontend.

    Parameters:
        y_true (list/array): actual intermittent demand
        y_pred (list/array): model forecast
        title (str): chart title

    Returns:
        dict: Plotly figure dictionary
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # X-axis labels as W1, W2, W3...
    x_labels = [f"W{i+1}" for i in range(len(y_true))]

    # --- Demand Trace (Green solid line + markers)
    demand_trace = go.Scatter(
        x=x_labels,
        y=y_true,
        mode="lines+markers",
        name="demand",
        line=dict(color="green", width=3),
        marker=dict(size=8, color="green")
    )

    # --- Forecast Trace (Blue dashed + markers)
    forecast_trace = go.Scatter(
        x=x_labels,
        y=y_pred,
        mode="lines+markers",
        name="forecast",
        line=dict(color="blue", width=2, dash="dash"),
        marker=dict(size=8, color="blue")
    )

    # --- Layout
    layout = go.Layout(
        title=title,
        template="plotly_dark",  # match your dark UI theme
        xaxis=dict(title="", showgrid=True, gridcolor="#444444"),
        yaxis=dict(title="", showgrid=True, gridcolor="#444444"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        margin=dict(l=40, r=20, t=60, b=80)
    )

    return {
        "data": [demand_trace, forecast_trace],
        "layout": layout
    }

import numpy as np

def get_zero_nonzero_periods(demand):
    """
    Analyze intermittent demand and return:
    - zero-demand indices
    - non-zero-demand indices
    - number of zeros
    - number of non-zeros
    - continuous zero-demand blocks (start-end)
    """

    demand = np.array(demand)
    n = len(demand)

    zero_idx = np.where(demand == 0)[0].tolist()
    nonzero_idx = np.where(demand > 0)[0].tolist()

    # Count
    zero_count = len(zero_idx)
    nonzero_count = len(nonzero_idx)

    # Find continuous zero-demand blocks (zero intervals)
    zero_blocks = []
    if zero_idx:
        start = zero_idx[0]
        prev = zero_idx[0]

        for i in zero_idx[1:]:
            if i == prev + 1:
                # still in the block
                prev = i
            else:
                zero_blocks.append((start, prev))
                start = i
                prev = i

        zero_blocks.append((start, prev))

    return {
        "zero_indices": zero_idx,
        "nonzero_indices": nonzero_idx,
        "zero_count": zero_count,
        "nonzero_count": nonzero_count,
        "total_periods": n,
        "zero_inflation_rate": round((zero_count / n) * 100, 2),
        "zero_blocks": zero_blocks
    }

import numpy as np

def get_realtime_adjustment(y_true, y_pred, window=7):
    """
    Measures real-time adjustment by comparing recent error vs older error.
    Returns value between 0 and 1.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) < window * 2:
        return None

    # Error right after demand shock (older)
    old_err = np.mean(np.abs(y_true[-2*window:-window] - y_pred[-2*window:-window]))

    # Current error (recent)
    new_err = np.mean(np.abs(y_true[-window:] - y_pred[-window:]))

    if old_err == 0:
        return 1.0

    score = 1 - (new_err / old_err)
    return float(round(max(min(score, 1), -1), 4))

from datetime import datetime, timezone

def get_data_freshness(last_timestamp, unit="hours"):
    """
    last_timestamp: datetime object
    Returns how old the data is (in hours or minutes or days).
    """

    now = datetime.now(timezone.utc)
    diff = now - last_timestamp

    if unit == "minutes":
        freshness = diff.total_seconds() / 60
    elif unit == "days":
        freshness = diff.total_seconds() / 86400
    else:
        freshness = diff.total_seconds() / 3600

    return float(round(freshness, 3))

import time

def measure_latency(func, *args, **kwargs):
    """
    Measures runtime of a function call.
    Returns:
        - output (what the function returns)
        - latency (seconds)
    """

    start = time.time()
    output = func(*args, **kwargs)
    end = time.time()

    latency = round(end - start, 4)

    return output, latency

def system_health_metrics(y_true, y_pred, last_data_ts, latency_sec):
    return {
        "realtime_adjustment": get_realtime_adjustment(y_true, y_pred),
        "data_freshness_hours": get_data_freshness(last_data_ts),
        "latency_seconds": round(latency_sec, 3)
    }

import numpy as np
import plotly.graph_objs as go

def get_baseline_vs_realtime_adjustment_graph(y_true, y_pred, window=7, title="Baseline vs Real-Time Adjustment"):
    """
    Creates an interactive Plotly graph comparing baseline error vs real-time error.
    Returns Plotly figure dict (JSON-serializable).

    Parameters:
        y_true : actual values
        y_pred : predicted values
        window : number of periods for real-time adaptation block
        title  : chart title

    Returns:
        dict: {
            "data": [...],
            "layout": {...}
        }
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n = len(y_true)

    if n < 2 * window:
        raise ValueError("Not enough data to compute baseline and real-time windows.")

    # --- Compute baseline & realtime error series ---
    baseline_true = y_true[-2*window:-window]
    baseline_pred = y_pred[-2*window:-window]
    baseline_error = np.abs(baseline_true - baseline_pred)

    recent_true = y_true[-window:]
    recent_pred = y_pred[-window:]
    realtime_error = np.abs(recent_true - recent_pred)

    # X-axis labels
    x_baseline = [f"t-{2*window-i}" for i in range(window)]
    x_realtime = [f"t-{window-i}" for i in range(window)]

    # --- Plotly Traces ---
    baseline_trace = go.Scatter(
        x=x_baseline,
        y=baseline_error,
        mode="lines+markers",
        name="Baseline Error",
        line=dict(color="orange", width=3),
        marker=dict(size=8, color="orange")
    )

    realtime_trace = go.Scatter(
        x=x_realtime,
        y=realtime_error,
        mode="lines+markers",
        name="Real-Time Error",
        line=dict(color="deepskyblue", width=3),
        marker=dict(size=8, color="deepskyblue")
    )

    # --- Layout ---
    layout = go.Layout(
        title=title,
        template="plotly_dark",
        xaxis=dict(title="Time Period", showgrid=True, gridcolor="#3d3d3d"),
        yaxis=dict(title="Absolute Error", showgrid=True, gridcolor="#3d3d3d"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        margin=dict(l=50, r=30, t=60, b=80),
    )

    return {
        "data": [baseline_trace, realtime_trace],
        "layout": layout
    }


