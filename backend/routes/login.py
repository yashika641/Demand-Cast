from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
import os

router = APIRouter(prefix="/auth", tags=["Auth"])

# ------------ MODELS ------------
class EmailPassword(BaseModel):
    email: str
    password: str


# ------------ SUPABASE INIT ------------
load_dotenv(r"C:\Users\palya\Desktop\demancast\backend\.env")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

PROVIDERS = ["google", "github"]


# ============================================
#  SIGNUP — EMAIL + PASSWORD
# ============================================
@router.post("/signup")
def signup_email(data: EmailPassword):
    try:
        result = supabase.auth.sign_up(
            {"email": data.email, "password": data.password}
        )
        return {
            "message": "Signup successful",
            "user": result.user
        }

    except Exception as e:
        raise HTTPException(400, detail=str(e))


# ============================================
#  LOGIN — EMAIL + PASSWORD
# ============================================
@router.post("/login")
def login_email(data: EmailPassword):
    try:
        result = supabase.auth.sign_in_with_password(
            {"email": data.email, "password": data.password}
        )

        if not result.user:
            raise HTTPException(401, "Invalid credentials")

        return {
            "message": "Login successful",
            "access_token": result.session.access_token,
            "refresh_token": result.session.refresh_token,
            "user": result.user
        }

    except Exception as e:
        raise HTTPException(400, detail=str(e))


# ============================================
#  OAUTH SIGNUP + LOGIN — GOOGLE / GITHUB
# ============================================
@router.get("/oauth/{provider}")
def oauth_login(provider: str):

    if provider not in PROVIDERS:
        raise HTTPException(400, "Provider not supported")

    # This MUST match your FRONTEND CALLBACK PAGE:
    redirect_url = "http://localhost:5173/auth/callback"

    try:
        result = supabase.auth.sign_in_with_oauth(
            {
                "provider": provider,
                "options": {"redirect_to": redirect_url}
            }
        )
        return {"url": result.url}

    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ============================================
#  PROTECTED USER PROFILE ROUTE
# ============================================
@router.get("/me")
def get_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(401, "Missing token")

    token = authorization.replace("Bearer ", "")

    try:
        user = supabase.auth.get_user(token)
        return user
    except:
        raise HTTPException(401, "Invalid or expired token")
