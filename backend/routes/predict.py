from fastapi import APIRouter, Depends, Request
from app.core.firebase_auth import verify_firebase_user
from app.pipelines.churn_predict import predict_churn

router = APIRouter()

@router.post("/predict/churn")
async def predict_route(request: Request, user=Depends(verify_firebase_user)):
    payload = await request.json()
    uid = user["uid"]
    return predict_churn(uid, payload["rows"])
