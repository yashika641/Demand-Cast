from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter

app = FastAPI(title="Demand-Cast api backend")
# Import route
from backend.routes import upload,login,signup
from backend.routes import sales
from backend.routes import google
from backend.routes import product
from backend.routes import customer
# from backend.routes.sales import route_sales_analytics, route_google_trends, route_revenue_analysis, route_segmentation_filtering, route_forecast_predictions
# Include route

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "http://localhost:5173",     # React dev server
    "http://127.0.0.1:5173","http://51.159.115.233:3128",
    "http://161.35.70.249:8080",
    "http://47.243.177.210:8080", ],  # or ["*"] for wide-open dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(login.router)
app.include_router(signup.router)
app.include_router(sales.router)
app.include_router(google.router)
app.include_router(product.router)
app.include_router(customer.router)
# app.include_router(route_sales_analytics)
# app.include_router(route_google_trends) 
# app.include_router(route_revenue_analysis)
# app.include_router(route_segmentation_filtering)
# app.include_router(route_forecast_predictions)

@app.get("/")
def read_root():
    return {"Hello": "World"}