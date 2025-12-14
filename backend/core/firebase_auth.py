from fastapi import Depends, HTTPException, Header
import firebase_admin
from firebase_admin import auth, credentials

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

async def verify_firebase_user(authorization: str = Header(...)):
    try:
        token = authorization.split(" ")[1]
        decoded = auth.verify_id_token(token)
        return decoded  # contains 'uid', 'email', etc.
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired Firebase token")
