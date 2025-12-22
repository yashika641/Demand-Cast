import numpy as np

def compute_stockout_risk_metrics(
    demand_forecast,      # dict: {sku: [forecast list]}
    lead_time,            # dict: {sku: lead_time_days}
    current_stock,        # dict: {sku: stock_level}
    stockout_proba,       # dict: {sku: model_probability}
    safety_factor=1.0,    # additional buffer multiplier
    risk_threshold=0.6    # high-risk threshold
):
    """
    Computes:
    - overall_stockout_risk
    - high_risk_skus
    - critical_alerts
    - avg_days_to_stockout

    Inputs come from:
    - demand forecast model
    - lead-time per SKU
    - stockout ML model probability
    """

    days_to_stockout = {}
    risk_scores = {}
    high_risk = {}
    critical_alerts = []

    for sku, forecast in demand_forecast.items():

        lt = lead_time.get(sku, 7)
        stock = current_stock.get(sku, 0)
        proba = stockout_proba.get(sku, 0)

        # ---- 1. Lead-time demand ----
        lt_demand = np.sum(forecast[:lt]) * safety_factor

        # ---- 2. Days to stockout (forecast-based) ----
        cumulative = 0
        dts = None
        for i, d in enumerate(forecast):
            cumulative += d
            if cumulative >= stock:
                dts = i + 1
                break
        if dts is None:
            dts = float("inf")  # will not stockout soon

        days_to_stockout[sku] = dts

        # ---- 3. Stockout Risk Score (probability × exposure) ----
        exposure = lt_demand / (stock + 1)
        risk = float(min(1, proba * exposure))
        risk_scores[sku] = risk

        # ---- 4. High-risk SKUs ----
        if risk >= risk_threshold:
            high_risk[sku] = risk

        # ---- 5. Critical alerts ----
        if dts <= 3 or stock <= 5:
            critical_alerts.append({
                "sku": sku,
                "stock": stock,
                "days_to_stockout": dts,
                "risk_score": float(round(risk, 3))
            })

    # ---- 6. Overall Stockout Risk % ----
    overall_risk_pct = round(
        (sum([1 for r in risk_scores.values() if r >= risk_threshold]) / len(risk_scores)) * 100,
        2
    )

    # ---- 7. Avg Days to Stockout ----
    numeric_dts = [v for v in days_to_stockout.values() if np.isfinite(v)]
    avg_dts = float(round(np.mean(numeric_dts), 2)) if numeric_dts else None

    return {
        "overall_stockout_risk": overall_risk_pct,
        "high_risk_skus": dict(sorted(high_risk.items(), key=lambda x: x[1], reverse=True)),
        "critical_alerts": critical_alerts,
        "average_days_to_stockout": avg_dts,
        "days_to_stockout": days_to_stockout,
        "risk_scores": risk_scores
    }

import numpy as np
import plotly.graph_objs as go

def get_7day_stockout_trend(stockout_proba_daily, smooth=True, title="7-Day Stockout Probability Trend"):
    """
    Creates a smooth 7-day stockout probability trend graph.
    
    stockout_proba_daily: list of daily stockout probabilities (0-1 range)
    smooth: apply smoothing filter
    """

    probs = np.array(stockout_proba_daily[:7]) * 100  # convert to %
    x_labels = [f"W{i+1}" for i in range(len(probs))]

    # ---- Smooth the curve if requested ----
    if smooth and len(probs) > 2:
        kernel = np.array([0.25, 0.5, 0.25])
        smoothed = np.convolve(probs, kernel, mode='same')
    else:
        smoothed = probs

    # ---- Line Plot (red gradient, like your UI) ----
    trace = go.Scatter(
        x=x_labels,
        y=smoothed,
        mode="lines",
        name="Stockout Probability",
        line=dict(color="red", width=4),
        fill="tozeroy",
        fillcolor="rgba(255,0,0,0.15)"
    )

    layout = go.Layout(
        title=title,
        template="plotly_dark",
        yaxis=dict(title="Probability (%)", range=[0, 100]),
        xaxis=dict(title=""),
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=50, r=30, t=60, b=80),
    )

    return {"data": [trace], "layout": layout}

import numpy as np
import plotly.graph_objs as go

def monte_carlo_demand_simulation(
    base_forecast, 
    lead_time_days,
    num_simulations=100,
    noise_factor=0.15,
    title="Monte Carlo Demand Simulation"
):
    """
    Simulates random demand paths based on forecast mean ± noise.
    """

    forecast = np.array(base_forecast[:lead_time_days])
    mu = forecast.mean()
    sigma = forecast.std()

    # ---- Monte Carlo Simulation Results ----
    simulations = []

    for _ in range(num_simulations):
        random_noise = np.random.normal(mu, sigma * noise_factor, size=len(forecast))
        sim_series = np.clip(forecast + random_noise, 0, None)  # no negative demand
        simulations.append(sim_series.sum())  # total demand over LT

    # ---- x-axis (sim run index) ----
    x = list(range(1, num_simulations+1))

    trace = go.Scatter(
        x=x,
        y=simulations,
        mode="markers",
        marker=dict(size=8, color="deepskyblue"),
        name="Simulated Demand"
    )

    layout = go.Layout(
        title=title,
        template="plotly_dark",
        xaxis=dict(title="Simulation Run"),
        yaxis=dict(title="Total Lead-Time Demand"),
        hovermode="closest",
        margin=dict(l=50, r=30, t=60, b=40),
    )

    return {"data": [trace], "layout": layout}

import pandas as pd
import numpy as np

def build_high_risk_sku_table(df, risk_threshold=0.45):
    """
    Builds High-Risk SKU Table directly from the final stockout DataFrame.

    Expected columns:
      - product_id
      - stock
      - safety_stock
      - stockout_proba (0–1)
      - days_to_stockout

    Returns a list of dicts → frontend-ready.
    """

    required_cols = [
        "product_id",
        "stock",
        "safety_stock",
        "stockout_proba",
        "days_to_stockout"
    ]
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Filter high-risk items
    high_risk = df[df["stockout_proba"] >= risk_threshold].copy()

    # Sort descending by risk
    high_risk = high_risk.sort_values("stockout_proba", ascending=False)

    # Build response table rows
    table = []
    for _, row in high_risk.iterrows():
        table.append({
            "sku": row["product_id"],
            "current_stock": int(row["stock"]),
            "safety_stock": int(row["safety_stock"]),
            "stockout_risk": f"{int(row['stockout_proba'] * 100)}%",
            "days_to_stockout": f"{int(row['days_to_stockout'])} days"
                if np.isfinite(row["days_to_stockout"]) else "N/A",
            "action": "Reorder"
        })

    return table

import pandas as pd
import numpy as np

def compute_supplier_reliability_metrics(df):
    """
    Computes all supplier KPIs:
      - avg supplier reliability (% on-time)
      - avg on-time delivery %
      - avg lead time
      - supplier-level table

    Expected df columns:
      supplier_id
      promised_lead_time
      actual_lead_time
      on_time (bool or 0/1)  ← computed inside if missing

    Returns a dict (JSON-ready)
    """

    required_cols = ["supplier_id", "promised_lead_time", "actual_lead_time"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # If on_time column doesn't exist → compute it
    if "on_time" not in df.columns:
        df["on_time"] = (df["actual_lead_time"] <= df["promised_lead_time"]).astype(int)

    # ---- Supplier-Level Reliability ----
    supplier_group = df.groupby("supplier_id")

    supplier_table = supplier_group.apply(lambda x: pd.Series({
        "avg_lead_time": float(round(x["actual_lead_time"].mean(), 2)),
        "on_time_delivery": float(round(x["on_time"].mean() * 100, 2)),
        "total_orders": int(len(x))
    })).reset_index()

    # ---- Overall KPIs ----
    avg_reliability = float(round(df["on_time"].mean() * 100, 2))
    avg_lead_time = float(round(df["actual_lead_time"].mean(), 2))

    # ---- Build Return Dict ----
    return {
        "avg_supplier_reliability": avg_reliability,  # % on time
        "avg_on_time_delivery": avg_reliability,      # alias
        "avg_lead_time": avg_lead_time,
        "suppliers": supplier_table.to_dict(orient="records")
    }

import plotly.graph_objs as go
import numpy as np
import pandas as pd

def supplier_performance_comparison_graph(supplier_df, title="Supplier Performance Comparison"):
    """
    Creates a Plotly bar graph comparing supplier performance metrics.

    Expected supplier_df columns:
        supplier_id        (string)
        on_time_delivery   (percentage 0–100)
        avg_lead_time      (in days)
        reliability_score  (optional, 0–100)

    Returns:
        dict: JSON-serializable Plotly figure
    """

    required_cols = ["supplier_id", "on_time_delivery", "avg_lead_time"]
    for col in required_cols:
        if col not in supplier_df.columns:
            raise ValueError(f"Missing required column: {col}")

    suppliers = supplier_df["supplier_id"].tolist()
    ontime = supplier_df["on_time_delivery"].tolist()
    leadtime = supplier_df["avg_lead_time"].tolist()

    # Optional: reliability score (default = on-time delivery)
    if "reliability_score" in supplier_df.columns:
        reliability = supplier_df["reliability_score"].tolist()
    else:
        reliability = ontime  # fallback

    # --- Bar Traces ---
    trace_ontime = go.Bar(
        x=suppliers,
        y=ontime,
        name="On-Time Delivery (%)",
        marker=dict(color="seagreen"),
        opacity=0.9
    )

    trace_leadtime = go.Bar(
        x=suppliers,
        y=leadtime,
        name="Avg Lead Time (Days)",
        marker=dict(color="royalblue"),
        opacity=0.9
    )

    trace_reliability = go.Bar(
        x=suppliers,
        y=reliability,
        name="Reliability Score (%)",
        marker=dict(color="goldenrod"),
        opacity=0.9
    )

    # --- Layout ---
    layout = go.Layout(
        title=title,
        template="plotly_dark",
        xaxis=dict(title="Supplier"),
        yaxis=dict(title="Performance Metrics"),
        barmode="group",  # side-by-side bars
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25)
    )

    return {"data": [trace_ontime, trace_leadtime, trace_reliability], "layout": layout}

import numpy as np
import pandas as pd

def build_detailed_supplier_metrics_table(df):
    """
    Generates the Detailed Supplier Metrics Table from supplier DataFrame.
    
    Expected columns in df:
        supplier_id        → supplier name/ID
        actual_lead_time   → numeric
        promised_lead_time → numeric
        on_time            → 0 or 1 (or will be computed automatically)

    Returns:
        List[dict] → frontend-ready structured table
    """

    # Required columns
    required_cols = ["supplier_id", "actual_lead_time", "promised_lead_time"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Compute on_time column if missing
    if "on_time" not in df.columns:
        df["on_time"] = (df["actual_lead_time"] <= df["promised_lead_time"]).astype(int)

    # Group by supplier
    group = df.groupby("supplier_id")

    table = []

    for supplier, subdf in group:
        # Reliability score (%)
        reliability = float(round(subdf["on_time"].mean() * 100, 2))

        # Average lead time
        avg_lt = float(round(subdf["actual_lead_time"].mean(), 2))

        # Lead time variance (standard deviation)
        lt_var = float(round(subdf["actual_lead_time"].std(ddof=0), 2))

        # Convert to strings for frontend formatting
        avg_lt_label = f"{int(round(avg_lt))} days"
        lt_var_label = f"±{lt_var} days"

        # Supplier status based on reliability score
        if reliability >= 95:
            status = "Excellent"
        elif reliability >= 90:
            status = "Good"
        elif reliability >= 80:
            status = "Fair"
        else:
            status = "Poor"

        table.append({
            "supplier": supplier,
            "reliability_score": f"{int(reliability)}%",
            "avg_lead_time": avg_lt_label,
            "lead_time_variance": lt_var_label,
            "status": status
        })

    # Sort by reliability descending
    table = sorted(table, key=lambda x: int(x["reliability_score"].replace("%", "")), reverse=True)

    return table

