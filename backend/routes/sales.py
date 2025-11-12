from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
from backend.routes.auth import verify_firebase_user
from supabase import create_client
import numpy as np
import json
import io

# Supabase info (unchanged)
SUPABASE_URL = "https://waryjyqdedzdrwhxzare.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndhcnlqeXFkZWR6ZHJ3aHh6YXJlIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTk5NDI5MSwiZXhwIjoyMDc3NTcwMjkxfQ.5M4RLa6o-Ii1MAXLdyUUhOYFQmUHAZEVE0xiM2SxkOc"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Common Column Definitions (unchanged) ---
POSSIBLE_SALES_COLS = [
    "sales","Sales","SALES",
    "revenue","Revenue","REVENUE",
    "amount","Amount","AMOUNT",
    "price","Price","PRICE",
    "total_amount","Total_Amount","TOTAL_AMOUNT"
]

POSSIBLE_DATE_COLS = [
    "date","Date","DATE",
    "order_date","Order_Date","ORDER_DATE",
    "orderDate","created_at","createdAt",
    "timestamp","Timestamp","TIMESTAMP",
]

POSSIBLE_PRODUCT_COLS = [
    "product","product_name","productname",
    "product description","product_title",
    "item","item_name","item description",
    "model","model_name",
]

POSSIBLE_CATEGORY_COLS = [
    "category","category_name","category_title","category_type",
    "subcategory","sub_category","sub_category_name",
    "main_category","department","division","section",
    "product_category","product_type","product_group","product_family",
    "line_of_business","market_segment","class","class_name","class_id",
]

def get_columns(df):
    # keep exact function/usage the same
    from frontend.utils.column_finder import column_finder
    date_col = column_finder(df, POSSIBLE_DATE_COLS)
    product_col = column_finder(df, POSSIBLE_PRODUCT_COLS)
    category_col = column_finder(df, POSSIBLE_CATEGORY_COLS)
    sales_col = column_finder(df, POSSIBLE_SALES_COLS)
    return date_col, product_col, category_col, sales_col

router = APIRouter(tags=["analytics"])

def to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj

def _aggregate_fallback(df: pd.DataFrame, date_col: str, sales_col: str, freq_label: str):
    """
    Fallback aggregation if sales_aggregation_over_time_dict is unavailable
    or to ensure labels/values are simple lists for Chart.js.
    """
    freq_map = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "M",
        "Yearly": "Y",
    }
    freq = freq_map.get(freq_label, "D")
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, sales_col])
    df = df.sort_values(date_col)
    agg = df.resample(freq, on=date_col)[sales_col].sum().reset_index()
    labels = agg[date_col].dt.strftime("%Y-%m-%d").tolist()
    values = agg[sales_col].astype(float).round(2).tolist()
    title = f"{freq_label} Sales Over Time"
    xlabel = freq_label[:-2] if freq_label.endswith("ly") else freq_label  # just a neat label
    if xlabel.lower() == "year": xlabel = "Year"
    if xlabel.lower() == "month": xlabel = "Month"
    if xlabel.lower() == "week": xlabel = "Week"
    if xlabel.lower() == "day": xlabel = "Day"
    return {
        "labels": labels,
        "values": values,
        "title": title,
        "xlabel": xlabel,
        "ylabel": "total_amount" if "total_amount" in df.columns else "Total Sales",
    }

@router.get("/user-sales-analytics")
async def user_sales_analytics(
    user = Depends(verify_firebase_user),
    frequency: str = Query("Daily", description="One of: Daily, Weekly, Monthly, Yearly")
):
    uid = user["uid"]

    result = supabase.table("files").select("*").eq("user_id", uid).execute()
    files = result.data if hasattr(result, "data") else []
    sales_files = [f for f in files if "sales" in f["filename"].lower()]
    if not sales_files:
        return JSONResponse({"status": "not_uploaded", "message": "No sales files uploaded."})

    # normalize frequency to title-case to keep your existing variable usage (e.g. "Daily")
    freq_label = (frequency or "Daily").strip().lower().capitalize()
    if freq_label not in {"Daily","Weekly","Monthly","Yearly"}:
        freq_label = "Daily"

    analytics_outputs = []
    for file_meta in sales_files:
        file_url = file_meta["url"]
        filename = file_meta["filename"]

        try:
            response = requests.get(file_url)
        except Exception:
            analytics_outputs.append({"filename": filename, "error": "Failed to download file."})
            continue

        if response.status_code != 200:
            analytics_outputs.append({"filename": filename, "error": "Failed to download file."})
            continue

        try:
            df = pd.read_csv(io.StringIO(response.text))
        except Exception:
            analytics_outputs.append({"filename": filename, "error": "Failed to read CSV."})
            continue

        if df.empty:
            analytics_outputs.append({"filename": filename, "error": "Empty CSV file."})
            continue

        date_col, product_col, category_col, sales_col = get_columns(df)
        if not sales_col or not date_col:
            analytics_outputs.append({"filename": filename, "error": "No recognized date/sales column in this file."})
            continue

        # --- stats (keep variable names) ---
        stats = {
            "total_sales": float(pd.to_numeric(df[sales_col], errors="coerce").fillna(0).sum()),
            "avg_order_value": float(pd.to_numeric(df[sales_col], errors="coerce").fillna(0).mean()),
            "row_count": int(len(df)),
        }

        # --- plot (keep variable names) ---
        # Try to use your existing module if present; otherwise fallback
        try:
            from backend.sales_data_aggregation import sales_aggregation_over_time_dict
            plot_dict = sales_aggregation_over_time_dict(df, date_col, sales_col, freq_label)
            # ensure shape is labels/values/title/xlabel/ylabel:
            # if your helper returns exactly that, great; otherwise normalize:
            if not isinstance(plot_dict, dict) or "labels" not in plot_dict or "values" not in plot_dict:
                plot_dict = _aggregate_fallback(df, date_col, sales_col, freq_label)
            else:
                # fill missing optional keys to satisfy frontend
                plot_dict.setdefault("title", f"{freq_label} Sales Over Time")
                plot_dict.setdefault("xlabel", freq_label[:-2] if freq_label.endswith("ly") else freq_label)
                plot_dict.setdefault("ylabel", "total_amount")
                # coerce to plain lists/numbers
                plot_dict["labels"] = [str(x) for x in list(plot_dict["labels"])]
                plot_dict["values"] = [float(x) for x in list(plot_dict["values"])]
        except Exception:
            plot_dict = _aggregate_fallback(df, date_col, sales_col, freq_label)

        # --- optional visuals kept commented to preserve your variable names ---
        # revenue_stats = {
        #     "total_revenue": float(df[sales_col].sum()),
        #     "avg_revenue": float(df[sales_col].mean()),
        # }
        # try:
        #     revenue_fig = px.line(df, x=date_col, y=sales_col, title="Revenue Over Time")
        #     revenue_plot = revenue_fig.to_dict()
        # except Exception:
        #     revenue_plot = {}

        # Build item (keep keys/variable names)
        analytics_outputs.append({
            "filename": filename,
            "stats": stats,
            "plot": plot_dict,
            # "revenue_stats": revenue_stats,
            # "revenue_plot": revenue_plot,
        })

    # Fix JSON serialization for all problematic objects (keep variable names)
    for item in analytics_outputs:
        for key in ["plot", "revenue_plot", "product_pie", "category_pie", "forecast_plot"]:
            if key in item and item[key]:
                item[key] = json.loads(json.dumps(item[key], default=to_serializable))

    return JSONResponse(
    json.loads(json.dumps({
        "status": "success",
        "n_sales_files": len(analytics_outputs),
        "files": analytics_outputs
    }, default=to_serializable))
)
    
def apply_chart_style(fig, title=None):
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="Arial, sans-serif", color="#333"),
            x=0.5,
            xanchor="center"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=12, color="#444"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#333")
        ),
        margin=dict(l=20, r=20, t=50, b=40)
    )
    return fig


def top_contributors(df, group_col, value_col, threshold=0.5):
    df_sorted = df.groupby(group_col, as_index=False)[value_col].sum()
    df_sorted = df_sorted.sort_values(value_col, ascending=False)
    df_sorted["cum_percent"] = df_sorted[value_col].cumsum() / df_sorted[value_col].sum()

    top_df = df_sorted[df_sorted["cum_percent"] <= threshold]

    others = df_sorted[df_sorted["cum_percent"] > threshold][value_col].sum()
    if others > 0:
        top_df = pd.concat([
            top_df,
            pd.DataFrame({group_col: ["Others"], value_col: [others]})
        ])
    return top_df


from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import requests, io, json
from datetime import datetime
from backend.routes.auth import verify_firebase_user
from supabase import create_client
# from backend.routes.sales_utils import get_columns, top_contributors  # adjust import if needed

# router = APIRouter(tags=["analytics"])

@router.get("/filtering-segmentation")
async def filtering_segmentation(
    user=Depends(verify_firebase_user),
    start_date: str = Query(None, description="Optional filter start date (YYYY-MM-DD)"),
    end_date: str = Query(None, description="Optional filter end date (YYYY-MM-DD)")
):
    uid = user["uid"]

    # ---- Fetch user's uploaded files ----
    result = supabase.table("files").select("*").eq("user_id", uid).execute()
    files = result.data if hasattr(result, "data") else []
    sales_files = [f for f in files if "sales" in f["filename"].lower()]
    if not sales_files:
        return JSONResponse({"status": "not_uploaded", "message": "No sales files uploaded."})

    # ---- Use latest file ----
    file_meta = sales_files[-1]
    filename = file_meta.get("filename")
    file_url = (
        file_meta.get("file_url")
        or file_meta.get("url")
        or file_meta.get("public_url")
        or file_meta.get("path")
    )
    if not file_url:
        return JSONResponse({"status": "error", "message": "No valid file URL found."})

    # ---- Read CSV ----
    try:
        response = requests.get(file_url)
        if response.status_code != 200:
            raise ValueError(f"Download failed ({response.status_code})")
        df = pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"CSV read failed: {e}"})

    if df.empty:
        return JSONResponse({"status": "error", "message": "Empty CSV file."})

    # ---- Detect Columns ----
    date_col, product_col, category_col, sales_col = get_columns(df)
    if not sales_col or not date_col:
        return JSONResponse({
            "status": "error",
            "message": "Missing essential columns.",
            "columns": df.columns.tolist()
        })

    # ---- Clean Data ----
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, sales_col])
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce").fillna(0)
    df[sales_col] = df[sales_col].replace([np.inf, -np.inf], 0)
    df[sales_col] = df[sales_col].clip(lower=0, upper=1e9)
    df = df.sort_values(date_col)

    # ---- Apply Date Filters ----
    if start_date:
        try:
            start_dt = pd.to_datetime(start_date)
            df = df[df[date_col] >= start_dt]
        except Exception:
            return JSONResponse({"status": "error", "message": f"Invalid start_date format: {start_date}"})

    if end_date:
        try:
            end_dt = pd.to_datetime(end_date)
            df = df[df[date_col] <= end_dt]
        except Exception:
            return JSONResponse({"status": "error", "message": f"Invalid end_date format: {end_date}"})

    # 🧩 Handle empty result after filtering
    if df.empty:
        return JSONResponse({
            "status": "empty",
            "message": f"No records found between {start_date or 'beginning'} and {end_date or 'today'}."
        })

    # ---- Generate Charts ----
    try:
        # 1️⃣ Line Chart
        df_line = df.groupby(date_col, as_index=False)[sales_col].sum()
        df_line[sales_col] = df_line[sales_col].replace([np.inf, -np.inf, np.nan], 0)
        line_chart_data = {
            "x": df_line[date_col].dt.strftime("%Y-%m-%d").tolist(),
            "y": df_line[sales_col].astype(float).round(2).tolist(),
            "name": "Sales"
        }

        # 2️⃣ Product Chart
        product_chart_data = []
        if product_col:
            df_product = top_contributors(df, product_col, sales_col, 0.5)
            df_product[sales_col] = df_product[sales_col].replace([np.inf, -np.inf, np.nan], 0)
            product_chart_data = df_product.to_dict(orient="records")

        # 3️⃣ Category Chart
        category_chart_data = []
        if category_col:
            df_category = top_contributors(df, category_col, sales_col, 0.5)
            df_category[sales_col] = df_category[sales_col].replace([np.inf, -np.inf, np.nan], 0)
            category_chart_data = df_category.to_dict(orient="records")

        # Debug Output
        print("\n==== DEBUG SNAPSHOT ====")
        print("Date Range:", start_date, "→", end_date)
        print("Line chart sample:", line_chart_data["x"][:5], line_chart_data["y"][:5])
        print("Product chart sample:", product_chart_data[:3])
        print("Category chart sample:", category_chart_data[:3])
        print("=========================\n")

        payload = {
            "status": "success",
            "file_used": filename,
            "date_range": {"start": start_date, "end": end_date},
            "line_chart": line_chart_data,
            "product_chart": product_chart_data,
            "category_chart": category_chart_data,
            "meta": {
                "date_col": date_col,
                "sales_col": sales_col,
                "product_col": product_col,
                "category_col": category_col,
            },
        }

        # Safe serialization
        # ---- Final Safe Serialization ----
        def to_serializable(obj):
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.bool_)):
                return bool(obj)
            return str(obj) if pd.isna(obj) else obj

        def sanitize_json(obj):
            if isinstance(obj, float) and not np.isfinite(obj):
                return 0.0
            elif isinstance(obj, list):
                return [sanitize_json(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: sanitize_json(v) for k, v in obj.items()}
            return obj

        safe_payload = json.loads(json.dumps(payload, default=to_serializable, allow_nan=True))
        safe_payload = sanitize_json(safe_payload)

        return JSONResponse(content=safe_payload)


    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Chart generation failed: {str(e)}"})

from prophet import Prophet
import xgboost as xgb

def to_serializable(x):
    if isinstance(x, (np.int64, np.int32)): return int(x)
    if isinstance(x, (np.float64, np.float32)): return float(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    return None if (isinstance(x, float) and not np.isfinite(x)) else x

def sanitize_json(obj):
    if isinstance(obj, float):
        return 0.0 if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.int64, np.int32)): return int(obj)
    if isinstance(obj, list): return [sanitize_json(v) for v in obj]
    if isinstance(obj, dict): return {k: sanitize_json(v) for k, v in obj.items()}
    return obj

def agg_to_freq_label(agg_unit: str):
    # Prophet frequency & Pandas resample codes
    if agg_unit == "months": 
        return "MS"  # Month Start
    return "D"       # default: days

def prepare_series(df, date_col, sales_col, freq_code):
    # ensure datetime + numeric & regularize index by freq
    s = df[[date_col, sales_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s[sales_col] = pd.to_numeric(s[sales_col], errors="coerce")
    s = s.dropna(subset=[date_col, sales_col])
    # resample to daily or monthly sum
    s = s.set_index(date_col).sort_index().resample(freq_code)[sales_col].sum().reset_index()
    s.columns = ["ds", "y"]
    return s

@router.get("/forecasts")
async def forecasts(
    user = Depends(verify_firebase_user),
    horizon: int = Query(30, ge=1, le=365, description="Forecast horizon (count of periods)"),
    unit: str = Query("days", regex="^(days|months)$", description="Horizon unit"),
):
    """
    Hybrid Prophet + XGBoost forecast. 
    - horizon: number of periods to forecast
    - unit: 'days' or 'months'
    Returns plot-ready dict for React Plotly.
    """
    uid = user["uid"]

    # 1) find latest sales file
    result = supabase.table("files").select("*").eq("user_id", uid).execute()
    files = result.data if hasattr(result, "data") else []
    sales_files = [f for f in files if "sales" in f.get("filename", "").lower()]
    if not sales_files:
        return JSONResponse({"status": "not_uploaded", "message": "No sales files uploaded."})

    file_meta = sales_files[-1]
    filename = file_meta.get("filename")
    file_url = (file_meta.get("file_url") or file_meta.get("url") or 
                file_meta.get("public_url") or file_meta.get("path"))
    if not file_url:
        return JSONResponse({"status":"error","message":"No valid file URL on record."})

    # 2) load CSV
    try:
        r = requests.get(file_url)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"CSV read failed: {e}"})

    if df.empty:
        return JSONResponse({"status": "error", "message": "Empty CSV file.", "file": filename})

    # 3) column discovery
    date_col, product_col, category_col, sales_col = get_columns(df)
    if not (date_col and sales_col):
        return JSONResponse({"status":"error","message":"Could not detect date/sales columns.",
                             "columns": df.columns.tolist()})

    # 4) aggregate to requested freq
    freq_code = agg_to_freq_label(unit)
    series = prepare_series(df, date_col, sales_col, freq_code)
    if len(series) < 20:
        return JSONResponse({"status":"error","message":"Not enough history for forecasting (min ~20 points).",
                             "n_points": int(len(series))})

    # 5) Prophet
    try:
        m = Prophet()
        m.fit(series)
        future = m.make_future_dataframe(periods=horizon, freq=freq_code, include_history=True)
        fcst = m.predict(future)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prophet failed: {e}")

    # 6) XGBoost on residuals (simple features)
    try:
        merged = series.merge(fcst[["ds","yhat"]], on="ds", how="left")
        merged["residual"] = merged["y"] - merged["yhat"]
        merged["lag1"] = merged["y"].shift(1)
        merged["lag7"] = merged["y"].shift(7) if freq_code == "D" else merged["y"].shift(1)
        merged["dayofweek"] = merged["ds"].dt.dayofweek
        merged["month"] = merged["ds"].dt.month
        merged = merged.dropna()

        if len(merged) >= 10:
            X = merged[["lag1","lag7","dayofweek","month"]]
            y = merged["residual"]
            model = xgb.XGBRegressor(
                objective="reg:squarederror", n_estimators=200, learning_rate=0.05, max_depth=4
            )
            model.fit(X, y)

            # future feature grid for forecast periods
            future_tail = fcst.tail(horizon).copy()
            # create naive lags from last observed y
            last_y = float(series["y"].iloc[-1])
            last_y7 = float(series["y"].iloc[-7]) if freq_code == "D" and len(series)>=7 else last_y
            F = pd.DataFrame({
                "lag1": [last_y]*horizon,
                "lag7": [last_y7]*horizon,
                "dayofweek": future_tail["ds"].dt.dayofweek.values,
                "month": future_tail["ds"].dt.month.values
            })
            residual_pred = model.predict(F)
            future_tail["hybrid"] = future_tail["yhat"] + residual_pred
        else:
            # fallback: prophet only
            future_tail = fcst.tail(horizon).copy()
            future_tail["hybrid"] = future_tail["yhat"]
    except Exception as e:
        # if XGB fails, fallback to Prophet-only
        future_tail = fcst.tail(horizon).copy()
        future_tail["hybrid"] = future_tail["yhat"]

    # 7) Build plot-ready dict
    history = {
        "x": series["ds"].dt.strftime("%Y-%m-%d").tolist(),
        "y": series["y"].astype(float).round(3).tolist(),
        "name": "History"
    }

    prophet_line = {
        "x": future_tail["ds"].dt.strftime("%Y-%m-%d").tolist(),
        "y": future_tail["yhat"].astype(float).round(3).tolist(),
        "name": "Prophet",
    }

    hybrid_line = {
        "x": future_tail["ds"].dt.strftime("%Y-%m-%d").tolist(),
        "y": future_tail["hybrid"].astype(float).round(3).tolist(),
        "name": "Hybrid",
    }

    ci = {}
    if {"yhat_lower","yhat_upper"}.issubset(fcst.columns):
        tail_ci = fcst.tail(horizon)
        ci = {
            "x": tail_ci["ds"].dt.strftime("%Y-%m-%d").tolist(),
            "lower": tail_ci["yhat_lower"].astype(float).round(3).tolist(),
            "upper": tail_ci["yhat_upper"].astype(float).round(3).tolist(),
        }

    payload = {
        "status": "success",
        "file_used": filename,
        "meta": {
            "date_col": date_col,
            "sales_col": sales_col,
            "freq": "monthly" if freq_code=="MS" else "daily",
            "horizon": int(horizon),
            "unit": unit
        },
        "plot": {
            "history": history,
            "forecast": {
                "prophet": prophet_line,
                "hybrid": hybrid_line,
                "ci": ci
            }
        },
        "table": pd.DataFrame({
            "date": hybrid_line["x"],
            "prophet": prophet_line["y"],
            "hybrid": hybrid_line["y"]
        }).tail(10).to_dict(orient="records")
    }

    safe = json.loads(json.dumps(payload, default=to_serializable))
    safe = sanitize_json(safe)
    return JSONResponse(content=safe)