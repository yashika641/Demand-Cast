from fastapi import APIRouter, Request
from supabase import create_client
import os

router = APIRouter()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")   # MUST be service role
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@router.get("/auth/callback")
def oauth_callback(request: Request):
    code = request.query_params.get("code")

    if not code:
        return {"error": "No code provided"}

    # EXCHANGE CODE FOR SESSION
    res = supabase.auth.exchange_code_for_session(code)

    if res.session:
        return {
            "access_token": res.session.access_token,
            "refresh_token": res.session.refresh_token
        }
    print(res)
    return {"error": "Failed to exchange code"}
