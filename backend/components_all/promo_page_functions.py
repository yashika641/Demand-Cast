import numpy as np
import pandas as pd

# -------------------------------
# 1. Price Elasticity of Demand
# -------------------------------
def price_elasticity_of_demand(price, quantity):
    """
    Computes price elasticity using log differences (best practice).

    price: list or array
    quantity: list or array

    Returns elasticity value.
    """
    price = np.array(price)
    quantity = np.array(quantity)

    # avoid division by zero
    valid = (price > 0) & (quantity > 0)
    price = price[valid]
    quantity = quantity[valid]

    # log-log elasticity (more robust)
    elasticity = np.polyfit(np.log(price), np.log(quantity), 1)[0]

    return float(round(elasticity, 4))


# -----------------------------------------
# 2. Optimal Revenue-Maximizing Price Point
# -----------------------------------------
def optimal_price_revenue(price_current, elasticity):
    """
    Computes revenue-maximizing price using elasticity formula.

    price_current: current price
    elasticity: already computed elasticity (negative value)

    Returns optimal price.
    """

    if elasticity >= -1:
        # revenue-maximizing price is infinite if demand is not elastic
        return None

    optimal_price = price_current * (elasticity / (1 + elasticity))
    return float(round(optimal_price, 2))


# -------------------------------
# 3. Cross Price Elasticity (CPE)
# -------------------------------
def cross_price_elasticity(price_other, quantity_this):
    """
    Computes cross-price elasticity:
        % change in Q_A / % change in P_B
    """
    price_other = np.array(price_other)
    quantity_this = np.array(quantity_this)

    valid = (price_other > 0) & (quantity_this > 0)
    price_other = price_other[valid]
    quantity_this = quantity_this[valid]

    cpe = np.polyfit(np.log(price_other), np.log(quantity_this), 1)[0]

    return float(round(cpe, 4))


# ------------------------------------------
# 4. Combined Price Optimization Function
# ------------------------------------------
def price_optimization_analysis(price, quantity, cross_price=None):
    """
    Returns:
        - elasticity
        - optimal price point
        - revenue-maximizing price
        - cross elasticity (optional)

    price: array
    quantity: array
    cross_price: another product's price (optional)
    """

    elasticity = price_elasticity_of_demand(price, quantity)
    current_price = price[-1]

    optimal_p = optimal_price_revenue(current_price, elasticity)

    analysis = {
        "elasticity": elasticity,
        "current_price": float(current_price),
        "optimal_revenue_price": optimal_p,
        "elasticity_type": (
            "Elastic" if elasticity < -1 else 
            "Unit Elastic" if -1 <= elasticity <= -0.9 else 
            "Inelastic"
        )
    }

    if cross_price is not None:
        analysis["cross_price_elasticity"] = cross_price_elasticity(cross_price, quantity)

    return analysis

import numpy as np
import plotly.graph_objs as go

def price_elasticity_curve_graph(price, quantity, elasticity, optimal_price=None, title="Price–Demand Elasticity Curve"):
    """
    Creates a Plotly graph of:
      - elasticity demand curve
      - current price demand point
      - optimal revenue-maximizing price point (optional)

    INPUTS:
        price: historical price list (e.g., [100,120,140,...])
        quantity: historical demand list
        elasticity: computed elasticity (negative, from your model)
        optimal_price: computed revenue-maximizing price (optional)

    RETURNS:
        {"data": [...], "layout": {...}} → JSON-safe for frontend
    """

    price = np.array(price)
    quantity = np.array(quantity)

    # Fit scale factor 'a' for Q = a * P^E
    # log(Q) = log(a) + E * log(P)
    log_a = np.log(quantity) - elasticity * np.log(price)
    a = np.exp(log_a.mean())

    # Generate a smooth price range for curve
    p_min, p_max = price.min()*0.8, price.max()*1.2
    curve_prices = np.linspace(p_min, p_max, 200)
    curve_demand = a * (curve_prices ** elasticity)

    # --- Demand Curve ---
    curve_trace = go.Scatter(
        x=curve_prices,
        y=curve_demand,
        mode="lines",
        name="Elasticity Curve",
        line=dict(color="dodgerblue", width=3)
    )

    # --- Current Price Point ---
    current_price = price[-1]
    current_qty = quantity[-1]

    current_trace = go.Scatter(
        x=[current_price],
        y=[current_qty],
        mode="markers+text",
        name="Current Price",
        marker=dict(color="orange", size=12),
        text=["Current Price"],
        textposition="top center"
    )

    # --- Optimal Price Point (Optional) ---
    optimal_trace = None
    if optimal_price is not None:
        optimal_qty = a * (optimal_price ** elasticity)
        optimal_trace = go.Scatter(
            x=[optimal_price],
            y=[optimal_qty],
            mode="markers+text",
            name="Optimal Price",
            marker=dict(color="green", size=12),
            text=["Optimal Price"],
            textposition="top center"
        )

    # Layout
    layout = go.Layout(
        title=title,
        template="plotly_dark",
        xaxis=dict(title="Price", showgrid=True, gridcolor="#444"),
        yaxis=dict(title="Demand", showgrid=True, gridcolor="#444"),
        hovermode="closest",
        margin=dict(l=50, r=30, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3)
    )

    traces = [curve_trace, current_trace]
    if optimal_trace:
        traces.append(optimal_trace)

    return {"data": traces, "layout": layout}


import pandas as pd
import numpy as np

def compute_promotion_analytics(df):
    """
    Computes:
      - avg uplift rate
      - incremental revenue
      - promo ROI
      - active promotions

    Expected columns:
        promo_flag (0/1)
        baseline_demand
        actual_demand
        unit_price
        promo_cost
        promo_start_date (optional)
        promo_end_date (optional)

    Returns a dict for frontend.
    """

    required_cols = [
        "promo_flag",
        "baseline_demand",
        "actual_demand",
        "unit_price",
        "promo_cost"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    promo_df = df[df["promo_flag"] == 1].copy()
    if promo_df.empty:
        return {
            "avg_uplift_rate": 0,
            "incremental_revenue": 0,
            "promo_roi": 0,
            "active_promotions": 0
        }

    # -------------------
    # 1. Uplift Rate
    # -------------------
    # uplift = (actual - baseline) / baseline
    promo_df["uplift_rate"] = (
        (promo_df["actual_demand"] - promo_df["baseline_demand"]) 
        / promo_df["baseline_demand"].replace(0, np.nan)
    ).fillna(0)

    avg_uplift_rate = float(round(promo_df["uplift_rate"].mean() * 100, 2))

    # -------------------
    # 2. Incremental Revenue
    # -------------------
    # incremental units * price
    promo_df["incremental_revenue"] = (
        promo_df["actual_demand"] - promo_df["baseline_demand"]
    ) * promo_df["unit_price"]

    total_incremental_revenue = float(round(promo_df["incremental_revenue"].sum(), 2))

    # -------------------
    # 3. Promotion ROI
    # -------------------
    # ROI = (incremental revenue - cost) / cost
    total_cost = promo_df["promo_cost"].sum()
    promo_roi = (
        (total_incremental_revenue - total_cost) / total_cost
        if total_cost > 0 else 0
    )
    promo_roi = float(round(promo_roi * 100, 2))  # % ROI

    # -------------------
    # 4. Active Promotions
    # -------------------
    if "promo_start_date" in df.columns and "promo_end_date" in df.columns:
        today = pd.Timestamp.today()
        active_promos = df[
            (df["promo_flag"] == 1)
            & (df["promo_start_date"] <= today)
            & (df["promo_end_date"] >= today)
        ].shape[0]
    else:
        active_promos = promo_df.shape[0]

    # -------------------
    # Final output
    # -------------------
    return {
        "avg_uplift_rate": avg_uplift_rate,       # %
        "incremental_revenue": total_incremental_revenue,
        "promo_roi": promo_roi,                  # %
        "active_promotions": int(active_promos)
    }

import plotly.graph_objs as go
import numpy as np

def promo_uplift_curve_graph(df, title="Promotion Uplift Curve"):
    """
    Creates a Plotly graph showing:
      - baseline demand
      - actual demand
      - uplift (difference)

    Required columns:
        baseline_demand
        actual_demand
        promo_name or sku (optional)

    Returns:
        dict → plotly graph for frontend
    """

    # X axis labels
    if "promo_name" in df.columns:
        labels = df["promo_name"].tolist()
    elif "sku" in df.columns:
        labels = df["sku"].tolist()
    else:
        labels = [f"Promo {i+1}" for i in range(len(df))]

    baseline = df["baseline_demand"].tolist()
    actual = df["actual_demand"].tolist()

    uplift = (df["actual_demand"] - df["baseline_demand"]).tolist()
    uplift_pct = (df["actual_demand"] - df["baseline_demand"]) / df["baseline_demand"].replace(0, np.nan)
    uplift_pct = (uplift_pct.fillna(0) * 100).tolist()

    # traces
    baseline_trace = go.Scatter(
        x=labels,
        y=baseline,
        mode="lines+markers",
        name="Baseline Demand",
        line=dict(color="gray", width=3)
    )

    actual_trace = go.Scatter(
        x=labels,
        y=actual,
        mode="lines+markers",
        name="Actual Demand",
        line=dict(color="dodgerblue", width=3)
    )

    uplift_trace = go.Bar(
        x=labels,
        y=uplift,
        name="Incremental Uplift",
        marker=dict(color="green")
    )

    layout = go.Layout(
        title=title,
        template="plotly_dark",
        xaxis=dict(title="Promotions"),
        yaxis=dict(title="Units"),
        hovermode="x unified",
        margin=dict(l=50, r=40, t=60, b=60),
        barmode="group"
    )

    return {"data": [baseline_trace, actual_trace, uplift_trace], "layout": layout}

import plotly.graph_objs as go
import numpy as np

def promo_roi_vs_discount_graph(df, title="ROI vs Discount Rate"):
    """
    Creates a Plotly graph for:
        discount_rate (x-axis)
        promo ROI % (y-axis)

    Required columns:
        discount_rate   (0-1 range)
        promo_cost
        baseline_demand
        actual_demand
        unit_price
    """

    # Compute incremental revenue
    df = df.copy()
    df["incremental_units"] = df["actual_demand"] - df["baseline_demand"]
    df["incremental_revenue"] = df["incremental_units"] * df["unit_price"]

    # ROI formula
    df["roi"] = np.where(
        df["promo_cost"] > 0,
        (df["incremental_revenue"] - df["promo_cost"]) / df["promo_cost"],
        0
    )

    discounts = (df["discount_rate"] * 100).tolist()
    roi_pct = (df["roi"] * 100).tolist()

    trace = go.Scatter(
        x=discounts,
        y=roi_pct,
        mode="lines+markers",
        name="ROI %",
        line=dict(color="orange", width=3)
    )

    layout = go.Layout(
        title=title,
        template="plotly_dark",
        xaxis=dict(title="Discount (%)", showgrid=True),
        yaxis=dict(title="ROI (%)", showgrid=True),
        hovermode="closest",
        margin=dict(l=50, r=40, t=60, b=60)
    )

    return {"data": [trace], "layout": layout}

import pandas as pd
import numpy as np

def build_recent_promotional_performance_table(df):
    """
    Builds the 'Recent Promotional Performance' table.

    Expected columns:
        campaign
        baseline_demand
        actual_demand
        unit_price
        promo_cost

    Returns:
        List[dict] — formatted rows for frontend UI.
    """

    required_cols = [
        "campaign",
        "baseline_demand",
        "actual_demand",
        "unit_price",
        "promo_cost"
    ]

    # Ensure required columns exist
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    table_rows = []

    for _, row in df.iterrows():

        baseline = row["baseline_demand"]
        actual = row["actual_demand"]

        # Uplift %
        uplift_pct = 0
        if baseline > 0:
            uplift_pct = ((actual - baseline) / baseline) * 100

        # Incremental units
        incremental_units = actual - baseline

        # Incremental revenue
        incremental_revenue = incremental_units * row["unit_price"]

        # ROI (incremental revenue - cost) / cost
        cost = row["promo_cost"]
        if cost > 0:
            roi = (incremental_revenue - cost) / cost
        else:
            roi = 0

        # Build formatted row
        table_rows.append({
            "campaign": row["campaign"],
            "baseline": int(baseline),
            "actual": int(actual),
            "uplift_pct": f"+{int(round(uplift_pct))}%",
            "incremental": int(round(incremental_units)),
            "roi": f"{round(roi + 1, 1)}x"   # ROI displayed like "2.8x"
        })

    return table_rows

import numpy as np
import pandas as pd

def calculate_inventory_kpis(
    inventory_df,
    demand_df,
    supplier_df,
    service_level=0.95,
    holding_cost_rate=0.20,   # 20% annual carrying cost
    ordering_cost=250         # cost per PO (EOQ)
):
    """
    Calculates all KPIs required for the Inventory Dashboard.

    inventory_df expected columns:
        product_id
        stock
        unit_cost

    demand_df expected columns:
        product_id
        daily_demand   (avg)
        demand_std     (std dev)
        lead_time_days

    supplier_df expected columns:
        supplier_id
        reliability_score (0-1)
        avg_lead_time

    Returns dashboard JSON dict.
    """

    # ===============================
    # 1. TOTAL INVENTORY VALUE
    # ===============================
    inventory_df["inventory_value"] = inventory_df["stock"] * inventory_df["unit_cost"]
    total_inventory_value = float(round(inventory_df["inventory_value"].sum(), 2))

    # ===============================
    # 2. DAYS OF INVENTORY (DOI)
    # ===============================
    total_stock_units = inventory_df["stock"].sum()
    total_daily_demand = demand_df["daily_demand"].sum()

    days_of_inventory = (
        total_stock_units / total_daily_demand 
        if total_daily_demand > 0 else None
    )
    if days_of_inventory:
        days_of_inventory = round(days_of_inventory, 1)

    # ===============================
    # 3. INVENTORY TURNS
    # ===============================
    # Inventory Turns = Annual Demand / Avg Inventory
    annual_demand = total_daily_demand * 365
    avg_inventory = total_stock_units / 2
    inventory_turns = round(annual_demand / avg_inventory, 1)

    # ===============================
    # 4. CARRYING COST
    # ===============================
    carrying_cost = round(total_inventory_value * holding_cost_rate, 2)

    # ===============================
    # 5. EOQ (Optimal Reorder Quantity)
    # ===============================
    # EOQ = sqrt((2 * D * S) / H)
    D = annual_demand
    S = ordering_cost
    H = inventory_df["unit_cost"].mean() * holding_cost_rate

    eoq = int(np.sqrt((2 * D * S) / H)) if H > 0 else None

    # ===============================
    # 6. SAFETY STOCK
    # ===============================
    # SS = Z * σ * sqrt(LT)
    Z = 1.65  # 95% service level
    demand_df["safety_stock"] = (
        Z * demand_df["demand_std"] * np.sqrt(demand_df["lead_time_days"])
    )

    safety_stock_total = int(demand_df["safety_stock"].sum())

    # ===============================
    # 7. Preferred Supplier (best reliability)
    # ===============================
    best_supplier = supplier_df.sort_values(
        ["reliability_score", "avg_lead_time"], 
        ascending=[False, True]
    ).iloc[0]

    preferred_supplier = {
        "supplier_id": best_supplier["supplier_id"],
        "reliability": f"{int(best_supplier['reliability_score']*100)}%",
        "lead_time": f"{int(best_supplier['avg_lead_time'])} days"
    }

    # ===============================
    # FINAL OUTPUT DICTIONARY
    # ===============================
    return {
        "total_inventory_value": total_inventory_value,
        "days_of_inventory": days_of_inventory,
        "inventory_turns": inventory_turns,
        "carrying_cost": carrying_cost,

        "reorder_quantity": eoq,
        "preferred_supplier": preferred_supplier,
        "safety_stock": safety_stock_total
    }

import numpy as np
import plotly.graph_objs as go

def eoq_analysis_graph(
    annual_demand,
    ordering_cost,
    holding_cost_per_unit,
    q_values=None
):
    """
    Creates EOQ cost curve for different order quantities.

    Inputs:
        annual_demand: D
        ordering_cost: S (per order)
        holding_cost_per_unit: H (per unit per year)
        q_values: manually defined order quantities (optional)

    Returns:
        Plotly dict for EOQ Chart
    """

    if q_values is None:
        q_values = np.arange(100, 900, 100)

    ordering_cost_curve = (annual_demand / q_values) * ordering_cost
    holding_cost_curve = (q_values / 2) * holding_cost_per_unit
    total_cost_curve = ordering_cost_curve + holding_cost_curve

    # Optimal EOQ
    eoq = int(np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit))

    # EOQ total cost
    eoq_total_cost = float((annual_demand / eoq) * ordering_cost + (eoq / 2) * holding_cost_per_unit)

    trace = go.Scatter(
        x=q_values,
        y=total_cost_curve,
        mode="lines+markers",
        name="Total Cost",
        line=dict(color="dodgerblue", width=3),
        marker=dict(size=8)
    )

    eoq_trace = go.Scatter(
        x=[eoq],
        y=[eoq_total_cost],
        mode="markers+text",
        name="Optimal EOQ",
        marker=dict(color="orange", size=12),
        text=[f"EOQ = {eoq}"],
        textposition="bottom center"
    )

    layout = go.Layout(
        title="Economic Order Quantity (EOQ) Analysis",
        template="plotly_dark",
        xaxis=dict(title="Order Quantity"),
        yaxis=dict(title="Total Cost ($)"),
        hovermode="closest",
        margin=dict(l=50, r=40, t=60, b=60)
    )

    return {"data": [trace, eoq_trace], "layout": layout, "eoq": eoq}

import plotly.graph_objs as go
import numpy as np

def inventory_level_tracking_graph(inventory_levels, labels=None, title="Inventory Level Tracking"):
    """
    Creates a weekly inventory level tracking graph.

    inventory_levels: list of inventory values [W1, W2, ..., W7]
    labels: week labels (optional)
    """

    if labels is None:
        labels = [f"W{i+1}" for i in range(len(inventory_levels))]

    trace = go.Scatter(
        x=labels,
        y=inventory_levels,
        mode="lines+markers",
        name="Inventory Level",
        line=dict(color="dodgerblue", width=4),
        fill="tozeroy",
        fillcolor="rgba(0, 136, 255, 0.15)",
        marker=dict(size=8)
    )

    layout = go.Layout(
        title=title,
        template="plotly_dark",
        xaxis=dict(title="Week"),
        yaxis=dict(title="Units"),
        hovermode="x unified",
        margin=dict(l=50, r=40, t=60, b=60)
    )

    return {"data": [trace], "layout": layout}
