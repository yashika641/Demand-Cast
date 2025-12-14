from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from supabase import create_client, Client
import pandas as pd
import numpy as np
import joblib
import io, os, json, hashlib, datetime, tempfile
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
import asyncio

router = APIRouter(prefix="/inventory", tags=["RETRAINING"])

load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)

# =====================================================================
# HELPERS — THESE WERE MISSING IN YOUR FILE (FIXED)
# =====================================================================

def get_user(token: str):
    """Authenticate user using Supabase token."""
    try:
        return supabase.auth.get_user(token).user
    except Exception:
        return None


def _normalize_supabase_download(res) -> bytes:
    """Return bytes from supabase.storage.download response."""
    if res is None:
        raise ValueError("Empty storage response")
    if isinstance(res, (bytes, bytearray)):
        return bytes(res)
    if hasattr(res, "content"):
        return res.content
    return bytes(res)


def read_file_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Reads CSV or XLSX from file bytes."""
    buf = io.BytesIO(file_bytes)
    try:
        buf.seek(0)
        return pd.read_csv(buf)
    except Exception:
        pass
    try:
        buf.seek(0)
        return pd.read_excel(buf)
    except Exception:
        pass
    raise ValueError("Unsupported file format")


def compute_hash(df1: pd.DataFrame, df2: pd.DataFrame) -> str:
    """Hash both datasets to detect change."""
    s1 = "" if df1 is None else df1.to_csv(index=False)
    s2 = "" if df2 is None else df2.to_csv(index=False)
    return hashlib.md5((s1 + s2).encode()).hexdigest()


def upload_model(uid: str, model_file: str):
    """Upload trained model to Supabase Storage."""
    filename = f"stockout_model_{uid}_{int(datetime.datetime.now().timestamp())}.pkl"
    storage_path = f"{uid}/{filename}"

    with open(model_file, "rb") as f:
        supabase.storage.from_("models").upload(
            storage_path, f, file_options={"upsert": True}
        )

    return storage_path, filename


# =====================================================================
# SSE FORMATTER
# =====================================================================
def sse(data: dict):
    return f"event: progress\ndata: {json.dumps(data)}\n\n"


# =====================================================================
# STREAM TRAINING
# =====================================================================
async def run_training(token: str):

    try:
        yield sse({"percent": 5, "message": "Authenticating user..."})

        # USER ======================================
        user = get_user(token)
        if not user:
            raise HTTPException(401, "Invalid token")

        uid = user.id

        yield sse({"percent": 10, "message": "Fetching user files..."})

        file_res = supabase.table("user_files").select("*").eq("user_id", uid).execute()
        rows = file_res.data or []

        if not rows:
            raise HTTPException(404, "User files not found")

        files = rows

        # --------------------------
        # DOWNLOAD functions
        # --------------------------
        def clean_path(url: str):
            if not url:
                return None
            if "/uploads/" in url:
                return url.split("/uploads/")[1]
            if "storage/v1/object/public/" in url:
                return url.split("storage/v1/object/public/")[-1]
            return url

        def safe_download(url: str):
            p = clean_path(url)
            r = supabase.storage.from_("uploads").download(p)
            b = _normalize_supabase_download(r)
            return read_file_bytes(b)

        # ======================================================
        # FIND FILES
        # ======================================================
        yield sse({"percent": 20, "message": "Loading sales & inventory..."})

        inventory_df = None
        sales_df = None

        for f in files:
            nm = (f.get("file_name") or f.get("file_url") or "").lower()
            if "inventory" in nm:
                inventory_df = safe_download(f.get("file_url"))
            if "sales" in nm:
                sales_df = safe_download(f.get("file_url"))

        if inventory_df is None or sales_df is None:
            raise HTTPException(400, "Missing sales or inventory file")

        # ======================================================
        # HASH CHECK
        # ======================================================
        yield sse({"percent": 30, "message": "Computing file hash..."})

        new_hash = compute_hash(inventory_df, sales_df)
        last_row = rows[0]
        old_hash = last_row.get("last_file_hash")

        if old_hash == new_hash and last_row.get("model_path"):
            yield sse({"percent": 100, "message": "Model already updated"})
            yield "data: DONE\n\n"
            return

        # ======================================================
        # PRODUCT COLUMN DETECTION
        # ======================================================
        yield sse({"percent": 40, "message": "Detecting product column..."})

        inv_cols = {c.lower(): c for c in inventory_df.columns}
        sales_cols = {c.lower(): c for c in sales_df.columns}

        product_candidates = [
            "sku", "product_id", "product", "item_id",
            "product_code", "item_code", "asin", "upc", "ean"
        ]

        product_column = None
        for c in product_candidates:
            if c in inv_cols and c in sales_cols:
                product_column = inv_cols[c]
                break

        if product_column is None:
            raise HTTPException(400, "Product column not found")

        # ======================================================
        # MERGE
        # ======================================================
        yield sse({"percent": 50, "message": "Merging datasets..."})
        df = sales_df.merge(inventory_df, on=product_column, how="left")

        # ======================================================
        # SALES & STOCK DETECTION
        # ======================================================
        yield sse({"percent": 60, "message": "Detecting stock/sales columns..."})

        merged_cols = {c.lower(): c for c in df.columns}

        sales_candidates = ["sales", "quantity", "qty", "units_sold", "sold_qty"]
        stock_candidates = ["stock", "current_stock", "inventory", "on_hand"]

        sales_column = next((merged_cols[c] for c in sales_candidates if c in merged_cols), None)
        stock_column = next((merged_cols[c] for c in stock_candidates if c in merged_cols), None)

        if sales_column is None:
            sales_column = df.select_dtypes(include=["int", "float"]).columns[0]
        if stock_column is None:
            stock_column = df.select_dtypes(include=["int", "float"]).columns[1]

        # ======================================================
        # FEATURES
        # ======================================================
        yield sse({"percent": 70, "message": "Creating features..."})

        df["stockout"] = (df[stock_column].fillna(0) <= 0).astype(int)

        X = df[[sales_column, stock_column]].fillna(0)
        y = df["stockout"]

        # ======================================================
        # TRAIN MODEL
        # ======================================================
        yield sse({"percent": 80, "message": "Training model..."})

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # ======================================================
        # SAVE MODEL
        # ======================================================
        yield sse({"percent": 90, "message": "Uploading model..."})

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
            model_file = tf.name

        joblib.dump(model, model_file)
        model_path, model_name = upload_model(uid, model_file)
        os.remove(model_file)

        # ======================================================
        # ANALYTICS
        # ======================================================
        yield sse({"percent": 95, "message": "Generating analytics..."})

        overall_stock_risk = float(df["stockout"].mean() * 100)
        avg_days_to_stockout = float(max(df[sales_column].mean(), 1))
        high_risk_skus = int((df[stock_column] < df[sales_column]).sum())
        critical_alerts = int((df[stock_column] == 0).sum())

        chart = (
            df.groupby(product_column)["stockout"]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        )

        # ======================================================
        # UPDATE DB
        # ======================================================
        yield sse({"percent": 98, "message": "Saving metadata..."})

        supabase.table("user_files").update({
            "model_path": model_path,
            "model_name": model_name,
            "last_file_hash": new_hash,
            "last_trained_at": datetime.datetime.utcnow().isoformat(),
            "overall_stock_risk": overall_stock_risk,
            "avg_days_to_stockout": avg_days_to_stockout,
            "high_risk_skus": high_risk_skus,
            "critical_alerts": critical_alerts,
            "stockout_chart": chart,
        }).eq("user_id", uid).execute()

        # ======================================================
        # DONE
        # ======================================================
        yield sse({"percent": 100, "message": "Training complete!"})
        yield "data: DONE\n\n"

    except Exception as e:
        yield sse({"percent": -1, "message": f"ERROR: {str(e)}"})
        yield "data: ERROR\n\n"


# =====================================================================
# STREAMING ENDPOINT
# =====================================================================
@router.post("/stockout/stream")
async def stream_stockout_training(Authorization: str = Header(None)):
    if not Authorization:
        raise HTTPException(401, "Missing token")

    token = Authorization.replace("Bearer ", "")

    async def event_generator():
        async for c in run_training(token):
            yield c
            await asyncio.sleep(0.01)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
