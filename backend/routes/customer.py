from backend.routes.auth import verify_firebase_user
import os, io, json, math, random, traceback
import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import JSONResponse
from supabase import create_client, Client

# ---------- Supabase setup ----------
SUPABASE_URL = "https://waryjyqdedzdrwhxzare.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndhcnlqeXFkZWR6ZHJ3aHh6YXJlIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTk5NDI5MSwiZXhwIjoyMDc3NTcwMjkxfQ.5M4RLa6o-Ii1MAXLdyUUhOYFQmUHAZEVE0xiM2SxkOc"
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

router = APIRouter(prefix="", tags=["customer_dashboard"])

# ---------- Utilities ----------
def column_finder(df: pd.DataFrame, candidates: list[str] | None) -> str | None:
    if df is None or df.empty or not candidates:
        return None
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def read_csv_from_url(url: str) -> pd.DataFrame:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")

def safe_json(obj):
    import math, json
    from datetime import datetime, date

    def convert(o):
        import numpy as np, pandas as pd
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return float(o) if math.isfinite(o) else 0.0
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (pd.Timestamp, datetime, date)):
            return o.isoformat()
        if isinstance(o, dict):
            return {str(k): convert(v) for k, v in o.items()}
        if isinstance(o, (list, tuple, set)):
            return [convert(v) for v in o]
        if o is None or (isinstance(o, float) and math.isnan(o)):
            return None
        return str(o)

    cleaned = convert(obj)
    return json.loads(json.dumps(cleaned, allow_nan=False))

# ---------- Candidate columns ----------
POSSIBLE_CUSTOMER_ID = ["customer_id", "Customer_ID", "cust_id", "id"]
POSSIBLE_GENDER = ["gender", "sex"]
POSSIBLE_AGE = ["age", "Age"]
POSSIBLE_REGION = ["region", "state", "city", "country"]
POSSIBLE_SIGNUP = ["signup_date", "sign_up", "joined", "join_date"]
POSSIBLE_MEMBERSHIP = ["membership_tier", "tier", "level"]
POSSIBLE_POINTS = ["loyalty_points", "points", "reward_points"]
POSSIBLE_FAV_ITEMS = ["highest_freq_purchased_items", "fav_item", "popular_item"]

# ---------- Route ----------
@router.get("/customer-dashboard")
async def customer_dashboard(user=Depends(verify_firebase_user)):
    uid = user["uid"]
    try:
        # --- Load files from Supabase ---
        files = []
        if supabase:
            res = supabase.table("files").select("*").eq("user_id", uid).execute()
            files = res.data if hasattr(res, "data") else []
        if not files:
            return JSONResponse({"status": "not_uploaded", "message": "No files found."})

        # Identify CSVs
        customer_file = next((f for f in files if "customer" in f["filename"].lower()), None)
        sales_file = next((f for f in files if "transaction" in f["filename"].lower()), None)
        if not customer_file:
            raise HTTPException(status_code=404, detail="Customer data not found.")

        # Load DataFrames
        def f_url(f): return f.get("file_url") or f.get("url")
        df_customers = read_csv_from_url(f_url(customer_file))
        df_sales = read_csv_from_url(f_url(sales_file)) if sales_file else pd.DataFrame()

        # Detect columns
        id_col = column_finder(df_customers, POSSIBLE_CUSTOMER_ID)
        gender_col = column_finder(df_customers, POSSIBLE_GENDER)
        age_col = column_finder(df_customers, POSSIBLE_AGE)
        region_col = column_finder(df_customers, POSSIBLE_REGION)
        signup_col = column_finder(df_customers, POSSIBLE_SIGNUP)
        member_col = column_finder(df_customers, POSSIBLE_MEMBERSHIP)
        points_col = column_finder(df_customers, POSSIBLE_POINTS)
        fav_item_col = column_finder(df_customers, POSSIBLE_FAV_ITEMS)

        # ---- DEMOGRAPHICS ----
        demographics = {}
        if gender_col:
            demographics["gender_distribution"] = df_customers[gender_col].value_counts(dropna=False).to_dict()
        if region_col:
            demographics["region_distribution"] = df_customers[region_col].value_counts(dropna=False).to_dict()
        if member_col:
            demographics["membership_tiers"] = df_customers[member_col].value_counts(dropna=False).to_dict()

        # ---- CLV ----
        clv_summary = {}
        if not df_sales.empty and id_col and id_col in df_sales.columns:
            amount_col = column_finder(df_sales, ["total_amount", "sales", "revenue", "order_value"])
            if amount_col:
                df_sales[amount_col] = pd.to_numeric(df_sales[amount_col], errors="coerce").fillna(0)
                clv = df_sales.groupby(id_col)[amount_col].sum().reset_index()
                clv.columns = [id_col, "CLV"]
                clv_summary["distribution"] = {
                    "mean": float(clv["CLV"].mean()),
                    "max": float(clv["CLV"].max()),
                    "min": float(clv["CLV"].min()),
                }
                clv_summary["top_customers"] = (
                    clv.sort_values("CLV", ascending=False).head(10).to_dict(orient="records")
                )

        # ---- LOYALTY ----
        loyalty_summary = {}
        if points_col:
            df_customers[points_col] = pd.to_numeric(df_customers[points_col], errors="coerce").fillna(0)
            loyalty_summary["avg_points"] = float(df_customers[points_col].mean())
            loyalty_summary["top_loyal_customers"] = (
                df_customers[[id_col, points_col]]
                .sort_values(points_col, ascending=False)
                .head(10)
                .to_dict(orient="records")
            )

        # ---- CHURN ----
        churn_summary = {"total_customers": len(df_customers)}
        if not df_sales.empty and id_col and id_col in df_sales.columns:
            date_col = column_finder(df_sales, ["date", "order_date", "purchase_date"])
            if date_col:
                df_sales[date_col] = pd.to_datetime(df_sales[date_col], errors="coerce")
                last_purchase = df_sales.groupby(id_col)[date_col].max().reset_index()
                last_purchase["days_since"] = (pd.Timestamp.today() - last_purchase[date_col]).dt.days
                churned = last_purchase[last_purchase["days_since"] > 180]
                churn_summary["churned_customers"] = len(churned)
                churn_summary["retention_rate"] = round(100 - (len(churned)/len(df_customers))*100, 2)

        # ---- SEGMENTATION ----
        segmentation = {}
        if clv_summary.get("top_customers"):
            clv = pd.DataFrame(clv_summary["top_customers"])
            clv["CLV_segment"] = pd.qcut(clv["CLV"], 3, labels=["Low Value", "Medium Value", "High Value"])
            segmentation["segments"] = clv["CLV_segment"].value_counts().to_dict()

        payload = {
            "status": "success",
            "demographics": demographics,
            "clv_summary": clv_summary,
            "loyalty_summary": loyalty_summary,
            "churn_summary": churn_summary,
            "segmentation": segmentation,
        }

        return JSONResponse(content=safe_json(payload))

    except Exception as e:
        print("=== CUSTOMER DASHBOARD ERROR ===")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in customer dashboard: {str(e)}")
