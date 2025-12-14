from fastapi import APIRouter, HTTPException
from backend.authutils import create_user  # assumes this exists and returns the new user's UID

router = APIRouter()

@router.post("/signup")
async def signup(email: str, password: str):
    """
    Server-driven signup using Firebase Admin SDK.
    Creates a new Firebase user and returns basic info.

    Note:
    - In typical flows, signup is performed on the frontend with the Firebase JS SDK,
      and the backend only verifies tokens for protected endpoints.
    - This route is useful for admin-style signup or migrating users.
    """
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")

    try:
        uid = create_user(email, password)  # should return the new user's UID
        return {"uid": uid, "email": email}
    except Exception as e:
        # Customize error handling as needed (map specific Firebase errors to HTTP status codes)
        raise HTTPException(status_code=400, detail=f"Signup failed: {str(e)}")
