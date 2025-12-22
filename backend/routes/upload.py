from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from supabase import create_client
from uuid import uuid4
import pandas as pd
import os
import io
import math
from dotenv import load_dotenv

load_dotenv(r"C:\Users\palya\Desktop\demancast\backend\.env")

router = APIRouter(prefix="/upload", tags=["upload"])

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ----------------------------------------------------------
# Utility: Extract user from JWT token
# ----------------------------------------------------------
def get_user(token: str):
    try:
        return supabase.auth.get_user(token).user
    except Exception:
        return None


# ----------------------------------------------------------
# Helper: Convert NaN / inf values to JSON-safe None
# ----------------------------------------------------------
def sanitize_df(df: pd.DataFrame):
    df = df.replace([float("inf"), float("-inf")], None)
    df = df.where(pd.notnull(df), None)
    return df


# ----------------------------------------------------------
# 🔵 ROUTE 1 — MULTI-FILE SCHEMA DETECTION
# ----------------------------------------------------------
@router.post("/schema-detect")
async def schema_detect(files: list[UploadFile] = File(...)):
    results = []

    for file in files:
        try:
            contents = await file.read()

            # Read file
            if file.filename.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(contents))
            elif file.filename.endswith(".xlsx"):
                df = pd.read_excel(io.BytesIO(contents))
            else:
                raise HTTPException(400, f"Unsupported file format: {file.filename}")

            # Sanitize
            df = sanitize_df(df)

            # Extract schema
            columns = [{"name": col, "type": str(df[col].dtype)} for col in df.columns]
            preview = df.head(10).to_dict(orient="records")

            results.append({
                "file_name": file.filename,
                "columns": columns,
                "preview": preview
            })

        except Exception as e:
            raise HTTPException(500, f"Error processing {file.filename}: {e}")

    return results


# ----------------------------------------------------------
# 🟡 ROUTE 2 — MULTI-FILE VALIDATION
# ----------------------------------------------------------
@router.post("/validate-data")
async def validate_data(payload: dict):
    files = payload.get("files", [])
    results = []

    for file in files:
        file_name = file["file_name"]
        preview = file["preview"]

        errors = []

        for i, row in enumerate(preview):
            for col, value in row.items():
                # Fix potential NaN/infinite
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    value = None

                # Negative demand
                if "demand" in col.lower() and isinstance(value, (float, int)):
                    if value is not None and value < 0:
                        errors.append({
                            "row": i + 1,
                            "column": col,
                            "issue": "Negative demand value"
                        })

                # Invalid date
                if "date" in col.lower():
                    try:
                        pd.to_datetime(value)
                    except:
                        errors.append({
                            "row": i + 1,
                            "column": col,
                            "issue": "Invalid date format"
                        })

        results.append({
            "file_name": file_name,
            "errors": errors
        })

    return {"results": results}


# ----------------------------------------------------------
# 🟢 ROUTE 3 — MULTI-FILE SUPABASE UPLOAD
# ----------------------------------------------------------
@router.post("/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    authorization: str = Header(None)
):

    if not authorization:
        raise HTTPException(401, "Missing Authorization header")

    token = authorization.replace("Bearer ", "")
    user = get_user(token)

    if user is None:
        raise HTTPException(401, "Invalid or expired token")

    user_id = user.id
    uploaded_files = []

    for file in files:
        try:
            file_bytes = await file.read()
            ext = file.filename.split(".")[-1]
            unique_name = f"{uuid4()}.{ext}"
            file_path = f"{user_id}/{unique_name}"

            # Upload to Supabase Storage
            supabase.storage.from_("uploads").upload(
                file_path,
                file_bytes,
                file_options={"content-type": file.content_type}
            )

            # Get public URL
            public_url = supabase.storage.from_("uploads").get_public_url(file_path)

            # Save metadata to DB
            db_res = supabase.table("user_files").insert({
                "user_id": user_id,
                "file_url": public_url,
                "file_name": file.filename
            }).execute()

            raw = db_res.model_dump()

            if raw.get("error"):
                raise HTTPException(500, f"DB insert failed: {raw['error']['message']}")

            uploaded_files.append({
                "original_name": file.filename,
                "stored_as": unique_name,
                "file_url": public_url
            })

        except Exception as e:
            raise HTTPException(500, f"Failed uploading {file.filename}: {e}")

    return {
        "message": "Files uploaded successfully",
        "uploaded_files": uploaded_files,
        "user_id": user_id
    }
