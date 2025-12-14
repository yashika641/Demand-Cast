# backend/routes/trends.py

from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import random
import time
from fake_useragent import UserAgent
import urllib3
from packaging import version

# Patch Retry object for urllib3>=2 compatibility
try:
    from urllib3.util import Retry
    if version.parse(urllib3.__version__) >= version.parse("2.0.0"):
        # Map the new attribute name for backward compatibility
        if not hasattr(Retry, "method_whitelist"):
            Retry.method_whitelist = Retry.allowed_methods
except Exception:
    pass

from pytrends.request import TrendReq

router = APIRouter(tags=["trends"])

# Random User-Agent generator
ua = UserAgent()

def create_safe_trend_client():
    """
    Create a TrendReq client with randomized headers and retry logic.
    Helps avoid 429 errors and improves reliability.
    """
    time.sleep(random.uniform(1.2, 3.0))  # small random delay
    headers = {"User-Agent": ua.random}
    return TrendReq(
        hl="en-US",
        tz=330,
        retries=5,
        backoff_factor=0.7,
        requests_args={"headers": headers},
    )

@router.get("/google-trends")
async def get_google_trends(
    product_name: str = Query(..., description="Product name or keyword to analyze"),
    timeframe: str = Query("today 12-m", description="e.g., 'today 5-y', 'today 12-m', 'now 7-d'"),
    geo: str = Query("", description="Optional region code like 'IN', 'US', etc.")
):
    """
    Fetch Google Trends data with anti-429 and version-safe retry handling.
    Uses randomized headers, smart backoff, and retry logic for reliability.
    """
    try:
        max_retries = 4

        for attempt in range(max_retries):
            try:
                pytrends = create_safe_trend_client()

                # Random human-like delay between requests
                time.sleep(random.uniform(1.0, 2.5))

                pytrends.build_payload(kw_list=[product_name], timeframe=timeframe, geo=geo)
                df = pytrends.interest_over_time()

                # If empty, retry with longer cooldown
                if df.empty:
                    if attempt < max_retries - 1:
                        print(f"⚠️ No data, retrying... (Attempt {attempt+1})")
                        time.sleep(random.uniform(2.0, 5.0))
                        continue
                    raise HTTPException(status_code=404, detail=f"No trend data found for '{product_name}'")

                # Clean and prepare DataFrame
                df.reset_index(inplace=True)
                df = df[["date", product_name]]
                df.columns = ["date", "interest"]
                df["date"] = pd.to_datetime(df["date"], errors="coerce").astype(str)
                df["interest"] = pd.to_numeric(df["interest"], errors="coerce").fillna(0)

                # Convert to clean JSON
                trend_json = df.to_dict(orient="records")

                return {
                    "status": "success",
                    "keyword": product_name,
                    "timeframe": timeframe,
                    "geo": geo or "global",
                    "trend_data": trend_json,
                    "n_points": len(trend_json),
                }

            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "too many requests" in error_str:
                    delay = random.uniform(4, 8)
                    print(f"🚨 429 rate limit detected. Cooling down for {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                elif attempt < max_retries - 1:
                    time.sleep(random.uniform(2.0, 4.0))
                    continue
                else:
                    raise

        raise HTTPException(status_code=500, detail="Failed to fetch Google Trends after retries.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend fetch failed: {str(e)}")
