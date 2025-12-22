import pandas as pd
import numpy as np
import os, json, io, re, asyncio
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
from supabase import create_client, Client
from dotenv import load_dotenv
from models.stockout.run_pipeline import run_stockout_pipeline
import base64
import mimetypes


# Load env
load_dotenv(r"C:\Users\palya\Desktop\demancast\backend\.env")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


# ----------------- HELPERS -----------------
def get_user(token: str):
    try:
        return supabase.auth.get_user(token).user
    except:
        return None

def sanitize_df(df):
    df = df.replace([float("inf"), float("-inf")], None)
    df = df.where(pd.notnull(df), None)
    return df

def _normalize_supabase_download(res):
    if res is None:
        raise ValueError("No data returned")
    if isinstance(res, (bytes, bytearray)):
        return bytes(res)
    if hasattr(res, "content"):
        return res.content
    return bytes(res)

def read_file_bytes(raw: bytes):
    buf = io.BytesIO(raw)
    try:
        return pd.read_csv(buf)
    except:
        buf.seek(0)
    try:
        return pd.read_excel(buf)
    except:
        raise ValueError("Unsupported file (CSV/Excel only)")


def sse_event(event: str, data: dict):
    """Format SSE event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ----------------- RETRIEVE FILES AS DATAFRAMES -----------------
def retrieve_files(token: str):
    user = get_user(token)
    if not user:
        return {"error": "Unauthorized"}, 401

    uid = user.id

    # ------------------ READ FROM user_files TABLE ------------------
    try:
        resp = (
            supabase.table("user_files")
            .select("*")
            .eq("user_id", uid)
            .order("uploaded_at", desc=True)
            .limit(50)
            .execute()
        )
        records = getattr(resp, "data", resp) or []
    except Exception as e:
        return {"error": f"Database lookup failed: {e}"}, 500

    if not records:
        return {"error": "No uploaded files found"}, 404

    # ---------------- REGEX TO MATCH FILE BY ORIGINAL NAME ----------------
    sales_rgx = re.compile(r"(sale|sales|order|transaction|demand)", re.I)
    inv_rgx   = re.compile(r"(inventory|stock|soh|onhand)", re.I)
    prod_rgx  = re.compile(r"(product|sku|item|catalog)", re.I)

    sales_df = inventory_df = products_df = None

    # Loop through all uploaded file records
    for rec in records:
        file_name = rec.get("file_name", "")  # original filename from user
        file_url  = rec.get("file_url", "")   # public Supabase URL

        if not file_url:
            continue

        # ---------------- MATCH FILE TYPE BY ORIGINAL NAME ----------------
        file_type = None
        if sales_rgx.search(file_name):
            file_type = "sales"
        elif inv_rgx.search(file_name):
            file_type = "inventory"
        elif prod_rgx.search(file_name):
            file_type = "products"
        else:
            continue  # skip irrelevant files

        # ---------------- CONVERT PUBLIC URL → STORAGE PATH ----------------
        # Example:
        # https://xyz.supabase.co/storage/v1/object/public/uploads/sales.csv
        # → uploads/sales.csv
        try:
            storage_path = file_url.split("/object/public/")[1]
        except:
            continue

        # If bucket is "uploads", remove prefix
        # "uploads/sales.csv" → "sales.csv"
        storage_relative = "/".join(storage_path.split("/")[1:])

        # ---------------- DOWNLOAD FILE ----------------
        try:
            dl = supabase.storage.from_("uploads").download(storage_relative)
            raw_bytes = _normalize_supabase_download(dl)
        except Exception as e:
            print("Download failed:", storage_relative, e)
            continue

        # ---------------- PARSE CSV/EXCEL → DATAFRAME ----------------
        try:
            df = sanitize_df(read_file_bytes(raw_bytes))
        except Exception as e:
            print("Parse failed:", file_name, e)
            continue

        # ---------------- ASSIGN TO CORRECT VARIABLE ----------------
        if file_type == "sales" and sales_df is None:
            sales_df = df

        elif file_type == "inventory" and inventory_df is None:
            inventory_df = df

        elif file_type == "products" and products_df is None:
            products_df = df

    # ---------------- RETURN ----------------
    return {
        "sales_df": sales_df,
        "inventory_df": inventory_df,
        "products_df": products_df
    }, 200




# ----------------- ROUTER -----------------
router = APIRouter(prefix="/inventory", tags=["inventory"])


@router.get("/")
async def status():
    return {"status": "ok"}





# ----------------- SSE ENDPOINT -----------------
# ----------------- SIMPLE PIPELINE RUNNER (NO SSE) -----------------
@router.post("/stockout")
async def run_stockout(request: Request):

    # 1️⃣ Extract token from Authorization header
    auth = request.headers.get("authorization")
    if not auth:
        raise HTTPException(401, "Token missing in headers")

    token = auth.replace("Bearer ", "").strip()
    # print("token", token)
    # 2️⃣ Validate user
    user = get_user(token)
    # print("user", user)
    if not user:
        raise HTTPException(401, "Invalid token")

    # 3️⃣ Retrieve files
    result, status = retrieve_files(token)

    # print("result", result)
    print("status", status)

    if status != 200:
        raise HTTPException(400, result.get("error"))

    # retrieve_files() returns already-parsed DataFrames
    sales_df = result.get("sales_df")
    inventory_df = result.get("inventory_df")
    products_df = result.get("products_df")

    print("sales_df", sales_df.head() if sales_df is not None else None)
    print("inventory_df", inventory_df.head() if inventory_df is not None else None)
    print("products_df", products_df.head() if products_df is not None else None)

    # 4️⃣ Validate required files
    if sales_df is None:
        raise HTTPException(400, "Sales file missing in storage")

    if inventory_df is None:
        raise HTTPException(400, "Inventory file missing in storage")

    # If products file missing → construct minimal DF
    if products_df is None:
        if "product_id" not in sales_df.columns:
            raise HTTPException(400, "Missing product_id column")
        
        products_df = pd.DataFrame({
            "product_id": sales_df["product_id"].unique()
        })

    # 4️⃣ Validate required files
    if sales_df is None:
        raise HTTPException(400, "Sales file missing in storage")
    if inventory_df is None:
        raise HTTPException(400, "Inventory file missing in storage")

    # If product file missing
    if products_df is None:
        if "product_id" not in sales_df.columns:
            raise HTTPException(400, "Missing product_id column")
        products_df = pd.DataFrame({"product_id": sales_df["product_id"].unique()})

    # 5️⃣ Run pipeline
    loop = asyncio.get_event_loop()
    try:
        output = await loop.run_in_executor(
            None,
            run_stockout_pipeline,
            sales_df,
            inventory_df,
            products_df,
        )
    except Exception as e:
        raise HTTPException(500, f"Pipeline failed: {str(e)}")
    print("output of pipeline",output)
    
    print("pipeline completed")
    # 6️⃣ Return JSON
    return {
        "kpis": output["metrics"],
        "stockout_trend": [],
        "monte_carlo": [],
        "high_risk_skus": [],
        "supplier_metrics": [],
        "anomaly_timeline": [],
        "data_quality_issues": [],
        "recommendations": [],
    }
