from fastapi import APIRouter, Depends
from app.core.firebase_auth import verify_firebase_user
from app.pipelines.churn_training import train_churn

router = APIRouter()

@router.post("/train/churn")
async def train_churn_model(user=Depends(verify_firebase_user)):
    uid = user["uid"]
    metrics, shap_summary = train_churn(uid)
    return {"status": "success", "metrics": metrics, "shap": shap_summary}
