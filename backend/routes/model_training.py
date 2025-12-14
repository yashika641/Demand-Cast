import os
from typing import Optional, List, Tuple
import re
import io
import json
import asyncio

from fastapi import FastAPI, Header, HTTPException, Depends, APIRouter
from pydantic import BaseModel
from redis import Redis
from rq import Queue, get_current_job
from supabase import create_client, Client
import pandas as pd
from dotenv import load_dotenv

from models.demand_forecast.run_pipeline import run_demand_pipeline
from models.leadtime.run_pipeline import run_leadtime_pipeline
from models.stockout.run_pipeline import run_stockout_pipeline
from models.promo_effectiveness_model.run_pipeline import run_promo_pipeline

# -----------------------------
# Supabase & Redis / RQ setup
# -----------------------------

load_dotenv(r"C:\Users\palya\Desktop\demancast\backend\.env")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # preferably service role
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase env vars not set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

REDIS_URL = "redis://localhost:6379/0"
redis_conn = Redis.from_url(REDIS_URL)
rq_queue = Queue("model_training", connection=redis_conn)


router = APIRouter(prefix="/model_training", tags=["model"])


# -----------------------------
# Helper: auth + users
# -----------------------------

def get_user(token: str):
    try:
        return supabase.auth.get_user(token).user
    except Exception:
        return None


def get_user_id(token: str):
    user = get_user(token)
    if user is None:
        raise HTTPException(401, "Invalid or expired token")
    return user.id


# -----------------------------
# Helper: dataframe sanitization
# -----------------------------

def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
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


def read_file_bytes(raw: bytes) -> pd.DataFrame:
    buf = io.BytesIO(raw)
    try:
        return pd.read_csv(buf)
    except Exception:
        buf.seek(0)
    try:
        return pd.read_excel(buf)
    except Exception:
        raise ValueError("Unsupported file (CSV/Excel only)")


# -----------------------------
# RETRIEVE FILES AS DATAFRAMES
# -----------------------------
def retrieve_files(token: str):
    user = get_user(token)
    if not user:
        return {"error": "Unauthorized"}, 401

    uid = user.id

    # Read from user_files table
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

    # Regex patterns for file type detection
    sales_rgx = re.compile(r"(sale|sales|order|transaction|demand)", re.I)
    inv_rgx = re.compile(r"(inventory|stock|soh|onhand)", re.I)
    prod_rgx = re.compile(r"(product|sku|item|catalog|catalogue)", re.I)
    customer_rgx = re.compile(r"(promo|promotion|pricing|campaigns)", re.I)
    supplier_rgx = re.compile(r"(suppliers|leadtime|supplier|vendor|supllier|supplier)", re.I)

    sales_df = None
    inventory_df = None
    products_df = None
    customer_df = None
    supplier_df = None

    # Loop through all uploaded file records
    for rec in records:
        file_name = rec.get("file_name", "")
        file_url = rec.get("file_url", "")

        if not file_url:
            continue

        # Determine file type from name
        file_type = None
        if sales_rgx.search(file_name):
            file_type = "sales"
        elif inv_rgx.search(file_name):
            file_type = "inventory"
        elif prod_rgx.search(file_name):
            file_type = "products"
        elif customer_rgx.search(file_name):
            file_type = "customer"
        elif supplier_rgx.search(file_name):
            file_type = "supplier"
        else:
            continue

        # Convert public URL -> storage path
        try:
            storage_path = file_url.split("/object/public/")[1]
        except Exception:
            continue

        storage_relative = "/".join(storage_path.split("/")[1:])

        # Download file
        try:
            dl = supabase.storage.from_("uploads").download(storage_relative)
            raw_bytes = _normalize_supabase_download(dl)
        except Exception as e:
            print("Download failed:", storage_relative, e)
            continue

        # Parse into dataframe
        try:
            df = sanitize_df(read_file_bytes(raw_bytes))
        except Exception as e:
            print("Parse failed:", file_name, e)
            continue

        # Assign to correct variable (first match wins)
        if file_type == "sales" and sales_df is None:
            sales_df = df
        elif file_type == "inventory" and inventory_df is None:
            inventory_df = df
        elif file_type == "products" and products_df is None:
            products_df = df
        elif file_type == "customer" and customer_df is None:
            customer_df = df
        elif file_type == "supplier" and supplier_df is None:
            supplier_df = df

    return {
        "sales_df": sales_df,
        "inventory_df": inventory_df,
        "products_df": products_df,
        "customer_df": customer_df,
        "supplier_df": supplier_df,
    }, 200


def get_files(result, status_code: int):
    if status_code == 200:
        sales_df = result.get("sales_df")
        inventory_df = result.get("inventory_df")
        products_df = result.get("products_df")
        customer_df = result.get("customer_df")
        supplier_df = result.get("supplier_df")

        print("sales_df", sales_df.head() if sales_df is not None else None)
        print("inventory_df", inventory_df.head() if inventory_df is not None else None)
        print("products_df", products_df.head() if products_df is not None else None)
        print("customer_df", customer_df.head() if customer_df is not None else None)
        print("supplier_df", supplier_df.head() if supplier_df is not None else None)

        if sales_df is None:
            raise HTTPException(400, "Sales file missing in storage")

        if inventory_df is None:
            raise HTTPException(400, "Inventory file missing in storage")

        if products_df is None:
            if "product_id" not in sales_df.columns:
                raise HTTPException(400, "Missing product_id column")
            products_df = pd.DataFrame({
                "product_id": sales_df["product_id"].unique()
            })

        if customer_df is None:
            raise HTTPException(400, "Customer file missing in storage")

        if supplier_df is None:
            raise HTTPException(400, "Supplier file missing in storage")

        return sales_df, inventory_df, products_df, customer_df, supplier_df
    else:
        print("no files found")
        raise HTTPException(status_code, "File retrieval failed")


def your_files(token: str):
    result, status_code = retrieve_files(token)
    sales_df, inventory_df, products_df, customer_df, supplier_df = get_files(
        result=result, status_code=status_code
    )
    return sales_df, inventory_df, products_df, customer_df, supplier_df


# -----------------------------
# Pydantic models (request body)
# -----------------------------

class DemandSettings(BaseModel):
    horizon: int
    model: str
    seasonality: str


class LeadtimeSettings(BaseModel):
    smoothing: int
    variability: bool = False


class StockoutSettings(BaseModel):
    balance: str
    model: str


class PromoSettings(BaseModel):
    lookback: int
    discount: bool


class PricingSettings(BaseModel):
    method: str
    cross: bool


class RagSettings(BaseModel):
    embedding: str
    dataset: str


class TrainingSettings(BaseModel):
    token: str  # <-- token from frontend body
    demand: DemandSettings
    leadtime: LeadtimeSettings
    stockout: StockoutSettings
    promo: PromoSettings
    pricing: PricingSettings
    rag: RagSettings


# -----------------------------
# Worker-side pipeline functions
# -----------------------------

def run_demand_pipeline_route(params: dict, token: str):
    job = get_current_job()
    job_id = job.id if job else None

    sales_df, inventory_df, products_df, customer_df, supplier_df = your_files(token)

    run_id = run_demand_pipeline(
        sales_df,
        horizon=params["horizon"],
        seasonality=params["seasonality"],
        model_type=params["model"],
    )

    if job_id:
        supabase.table("models").update({
            "run_id": run_id,
            "status": "completed",
        }).eq("job_id", job_id).execute()

    return run_id


def run_leadtime_pipeline_route(params: dict, token: str):
    job = get_current_job()
    job_id = job.id if job else None

    sales_df, inventory_df, products_df, customer_df, supplier_df = your_files(token)

    run_id = run_leadtime_pipeline(
        sales_df,
        inventory_df,
        supplier_df,
        smoothing=params["smoothing"],
        variability=params["variability"],
    )

    if job_id:
        supabase.table("models").update({
            "run_id": run_id,
            "status": "completed",
        }).eq("job_id", job_id).execute()

    return run_id


def run_stockout_pipeline_route(params: dict, token: str):
    job = get_current_job()
    job_id = job.id if job else None

    sales_df, inventory_df, products_df, customer_df, supplier_df = your_files(token)

    run_id = run_stockout_pipeline(
        sales_df,
        inventory_df,
        products_df,
        class_balancing=params["balance"],
        model_type=params["model"],
    )

    if job_id:
        supabase.table("models").update({
            "run_id": run_id,
            "status": "completed",
        }).eq("job_id", job_id).execute()

    return run_id


def run_promo_pipeline_route(params: dict, token: str):
    job = get_current_job()
    job_id = job.id if job else None

    sales_df, inventory_df, products_df, customer_df, supplier_df = your_files(token)

    run_id = run_promo_pipeline(
        trans_df=sales_df,
        promo_df=customer_df,
        lookback=params["lookback"],
        discount_sensitivity=params["discount"],
    )

    if job_id:
        supabase.table("models").update({
            "run_id": run_id,
            "status": "completed",
        }).eq("job_id", job_id).execute()

    return run_id


def run_pricing_pipeline(params: dict, token: str) -> str:
    # Placeholder
    return "pricing_run_id_example"


def run_rag_pipeline(params: dict, token: str) -> str:
    # Placeholder
    return "rag_run_id_example"


# -----------------------------
# Auth helper: get user from header token
# -----------------------------

def get_user_from_token(authorization: str = Header(...)) -> str:
    """
    Extracts Supabase user from JWT (Authorization: Bearer <token>)
    and returns user_id (UUID as string).
    """
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.split(" ", 1)[1].strip()

    try:
        auth_resp = supabase.auth.get_user(token)
        user = auth_resp.user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    if not user or not user.id:
        raise HTTPException(status_code=401, detail="User not found")

    return str(user.id)


# -----------------------------
# Main route: /start/train
# -----------------------------

@router.post("/start/train")
def start_train(settings: TrainingSettings, user_id: str = Depends(get_user_from_token)):
    """
    Takes in frontend model settings, enqueues training jobs via RQ
    for each model type, and stores (user_id, model_name, job_id, run_id)
    in Supabase 'models' table. MLflow run_id is filled from workers.
    """
    print("🔴 START_TRAIN HIT")

    # Quick Redis sanity check
    import redis
    try:
        r = redis.Redis(host="127.0.0.1", port=6379, db=0)
        print("Redis ping in route:", r.ping())
    except Exception as e:
        print("❌ Redis ping failed:", e)
        raise HTTPException(status_code=500, detail=f"Redis connection failed: {e}")

    try:
        # -----------------------
        # MAIN LOGIC WRAPPED HERE
        # -----------------------
        token = settings.token

        jobs: List[Tuple[str, str]] = []  # (model_name, job_id)

        # Demand
        print("➡ Enqueue demand job...")
        job_demand = rq_queue.enqueue(
            run_demand_pipeline_route,
            settings.demand.model_dump(),
            token,
            job_timeout=3600,
        )
        jobs.append(("demand", job_demand.id))

        # Leadtime
        print("➡ Enqueue leadtime job...")
        job_leadtime = rq_queue.enqueue(
            run_leadtime_pipeline_route,
            settings.leadtime.model_dump(),
            token,
            job_timeout="1h",
        )
        jobs.append(("leadtime", job_leadtime.id))

        # Stockout
        print("➡ Enqueue stockout job...")
        job_stockout = rq_queue.enqueue(
            run_stockout_pipeline_route,
            settings.stockout.model_dump(),
            token,
            job_timeout=3600
        )
        jobs.append(("stockout", job_stockout.id))

        # Promo
        print("➡ Enqueue promo job...")
        job_promo = rq_queue.enqueue(
            run_promo_pipeline_route,
            settings.promo.model_dump(),
            token,
            job_timeout=3600,
        )
        jobs.append(("promo", job_promo.id))

        # Insert job rows into 'models' table
        rows_to_insert = [
            {
                "user_id": user_id,
                "model_name": model_name,
                "job_id": job_id,
                "run_id": None,  # MLflow run id will be updated by worker
                "status": "queued",
            }
            for (model_name, job_id) in jobs
        ]

        print("➡ Inserting rows into Supabase:", rows_to_insert)

        supabase.table("models").insert(rows_to_insert).execute()

        estimated_time_minutes = 32

        print("✅ start_train completed OK")

        return {
            "message": "Training started",
            "estimated_time": estimated_time_minutes,
            "jobs": [
                {"model_name": model_name, "job_id": job_id}
                for (model_name, job_id) in jobs
            ],
        }

    except Exception as e:
        import traceback
        print("❌ Exception in start_train:")
        traceback.print_exc()
        # Surface the error to frontend for now (you can hide later)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
