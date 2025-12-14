# backend/routes/login.py

from fastapi import APIRouter, Header, HTTPException
from backend.authutils import verify_id_token

router = APIRouter()

@router.post("/login")
async def login(authorization: str = Header(None)):
    """
    Verifies Firebase ID token sent from frontend (React) after Firebase login.

    Expected header:
        Authorization: Bearer <Firebase_ID_Token>

    Returns:
        JSON with user's UID and email if verification succeeds.
    """
    # ✅ Step 1: Validate header
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header. Expected 'Bearer <token>'."
        )

    # ✅ Step 2: Extract token string
    id_token = authorization.split("Bearer ")[1]

    # ✅ Step 3: Verify token using Firebase Admin SDK
    try:
        user_info = verify_id_token(id_token)
        return {
            "uid": user_info["uid"],
            "email": user_info.get("email"),
            "message": "User verified successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Token invalid or expired: {str(e)}"
        )
