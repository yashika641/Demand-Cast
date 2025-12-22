# =============================
# Standard library imports
# =============================
import os
import re
import io
from typing import List, Tuple

# =============================
# Third-party imports
# =============================
import pandas as pd
import redis
from dotenv import load_dotenv
from fastapi import APIRouter, Header, HTTPException, Depends
from pydantic import BaseModel
from rq import Queue, get_current_job
from supabase import create_client, Client

# =============================
# Load env (LOCAL ONLY)
# Safe on Render (ignored if file not present)
# =============================
load_dotenv(r"C:\Users\palya\Desktop\demancast\Demand-Cast\backend\.env")

# =============================
# Supabase setup
# =============================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase environment variables not set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =============================
# Redis / RQ helpers (Upstash safe)
# =============================
def get_redis_connection():
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise RuntimeError("REDIS_URL environment variable not set")

    return redis.from_url(
        redis_url,
        ssl_cert_reqs=None  # REQUIRED for Upstash TLS
    )


def get_rq_queue():
    return Queue(
        "model_training",
        connection=get_redis_connection()
    )

# =============================
# Router
# =============================
router = APIRouter(
    prefix="/model_training",
    tags=["model"]
)

# =============================
# ML pipeline imports
# =============================
from backend.models.demand_forecast.run_pipeline import run_demand_pipeline
from models.leadtime.run_pipeline import run_leadtime_pipeline
from models.stockout.run_pipeline import run_stockout_pipeline
from models.promo_effectiveness_model.run_pipeline import run_promo_pipeline

# =============================
# Auth helpers
# =============================
def get_user(token: str):
    try:
        return supabase.auth.get_user(token).user
    except Exception:
        return None


def get_user_from_token(authorization: str = Header(...)) -> str:
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.split(" ", 1)[1].strip()

    try:
        user = supabase.auth.get_user(token).user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    if not user or not user.id:
        raise HTTPException(status_code=401, detail="User not found")

    return str(user.id)

# =============================
# Data helpers
# =============================
def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([float("inf"), float("-inf")], None)
    return df.where(pd.notnull(df), None)


def read_file_bytes(raw: bytes) -> pd.DataFrame:
    buf = io.BytesIO(raw)
    try:
        return pd.read_csv(buf)
    except Exception:
        buf.seek(0)
        return pd.read_excel(buf)

# =============================
# File retrieval from Supabase
# =============================
def retrieve_files(token: str):
    user = get_user(token)
    if not user:
        raise HTTPException(401, "Unauthorized")

    uid = user.id

    resp = (
        supabase.table("user_files")
        .select("*")
        .eq("user_id", uid)
        .order("uploaded_at", desc=True)
        .limit(50)
        .execute()
    )

    records = resp.data or []
    if not records:
        raise HTTPException(404, "No uploaded files found")

    patterns = {
        "sales": re.compile(r"(sale|sales|order|transaction|demand)", re.I),
        "inventory": re.compile(r"(inventory|stock|soh|onhand)", re.I),
        "products": re.compile(r"(product|sku|item|catalog)", re.I),
        "customer": re.compile(r"(promo|promotion|pricing|campaign)", re.I),
        "supplier": re.compile(r"(supplier|leadtime|vendor)", re.I),
    }

    dfs = {k: None for k in patterns}

    for rec in records:
        name = rec.get("file_name", "")
        url = rec.get("file_url", "")
        if not url:
            continue

        file_type = next(
            (k for k, rgx in patterns.items() if rgx.search(name)),
            None
        )
        if not file_type or dfs[file_type] is not None:
            continue

        try:
            path = url.split("/object/public/")[1]
            rel = "/".join(path.split("/")[1:])
            raw = supabase.storage.from_("uploads").download(rel)
            dfs[file_type] = sanitize_df(read_file_bytes(raw))
        except Exception:
            continue

    if dfs["sales"] is None:
        raise HTTPException(400, "Sales file missing")
    if dfs["inventory"] is None:
        raise HTTPException(400, "Inventory file missing")
    if dfs["customer"] is None:
        raise HTTPException(400, "Customer file missing")
    if dfs["supplier"] is None:
        raise HTTPException(400, "Supplier file missing")

    if dfs["products"] is None:
        dfs["products"] = pd.DataFrame({
            "product_id": dfs["sales"]["product_id"].unique()
        })

    return (
        dfs["sales"],
        dfs["inventory"],
        dfs["products"],
        dfs["customer"],
        dfs["supplier"],
    )

# =============================
# Request models
# =============================
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
    token: str
    demand: DemandSettings
    leadtime: LeadtimeSettings
    stockout: StockoutSettings
    promo: PromoSettings
    pricing: PricingSettings
    rag: RagSettings

# =============================
# Worker functions
# =============================
def run_demand_pipeline_route(params: dict, token: str):
    job = get_current_job()
    sales, inv, prod, cust, supp = retrieve_files(token)

    run_id = run_demand_pipeline(
        sales,
        horizon=params["horizon"],
        seasonality=params["seasonality"],
        model_type=params["model"],
    )

    if job:
        supabase.table("models").update(
            {"run_id": run_id, "status": "completed"}
        ).eq("job_id", job.id).execute()

    return run_id


def run_leadtime_pipeline_route(params: dict, token: str):
    job = get_current_job()
    sales, inv, prod, cust, supp = retrieve_files(token)

    run_id = run_leadtime_pipeline(
        sales,
        inv,
        supp,
        smoothing=params["smoothing"],
        variability=params["variability"],
    )

    if job:
        supabase.table("models").update(
            {"run_id": run_id, "status": "completed"}
        ).eq("job_id", job.id).execute()

    return run_id


def run_stockout_pipeline_route(params: dict, token: str):
    job = get_current_job()
    sales, inv, prod, cust, supp = retrieve_files(token)

    run_id = run_stockout_pipeline(
        sales,
        inv,
        prod,
        class_balancing=params["balance"],
        model_type=params["model"],
    )

    if job:
        supabase.table("models").update(
            {"run_id": run_id, "status": "completed"}
        ).eq("job_id", job.id).execute()

    return run_id


def run_promo_pipeline_route(params: dict, token: str):
    job = get_current_job()
    sales, inv, prod, cust, supp = retrieve_files(token)

    run_id = run_promo_pipeline(
        trans_df=sales,
        promo_df=cust,
        lookback=params["lookback"],
        discount_sensitivity=params["discount"],
    )

    if job:
        supabase.table("models").update(
            {"run_id": run_id, "status": "completed"}
        ).eq("job_id", job.id).execute()

    return run_id

# =============================
# Main API route
# =============================
@router.post("/start/train")
def start_train(
    settings: TrainingSettings,
    user_id: str = Depends(get_user_from_token),
):
    print("🔴 START_TRAIN HIT")

    # Redis sanity check
    try:
        get_redis_connection().ping()
    except Exception:
        raise HTTPException(500, "Redis connection failed")

    q = get_rq_queue()
    jobs: List[Tuple[str, str]] = []

    jobs.append(("demand", q.enqueue(
        run_demand_pipeline_route,
        settings.demand.model_dump(),
        settings.token,
        job_timeout=3600
    ).id))

    jobs.append(("leadtime", q.enqueue(
        run_leadtime_pipeline_route,
        settings.leadtime.model_dump(),
        settings.token,
        job_timeout=3600
    ).id))

    jobs.append(("stockout", q.enqueue(
        run_stockout_pipeline_route,
        settings.stockout.model_dump(),
        settings.token,
        job_timeout=3600
    ).id))

    jobs.append(("promo", q.enqueue(
        run_promo_pipeline_route,
        settings.promo.model_dump(),
        settings.token,
        job_timeout=3600
    ).id))

    supabase.table("models").insert([
        {
            "user_id": user_id,
            "model_name": name,
            "job_id": jid,
            "run_id": None,
            "status": "queued",
        }
        for name, jid in jobs
    ]).execute()

    return {
        "message": "Training started",
        "estimated_time": 32,
        "jobs": [
            {"model_name": name, "job_id": jid}
            for name, jid in jobs
        ],
    }
