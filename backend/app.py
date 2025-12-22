import fastapi
import fastapi.middleware.cors
from fastapi.middleware.cors import CORSMiddleware
from .routes.login import router as login_router
from .routes.authcallback import router as authcallback_router
from .routes.upload import router as upload_router
# from .routes.retraining import router as retrain_router
from .routes.risk_and_reliabilty import router as risk_router
from .routes.chatbot import router as chatbot_router
from .routes.model_training import router as model_training_router


app = fastapi.FastAPI(title="My FastAPI Application")

# origins = [
#     "http://localhost:5173",
#     "http://127.0.0.1:5173",
# ]
app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://dreamy-frangollo-57d87c.netlify.app/"
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(login_router)
app.include_router(upload_router)
# app.include_router(retrain_router)
app.include_router(risk_router)
app.include_router(chatbot_router)
app.include_router(model_training_router)

@app.get("/")
async def health_check():
    return {"status": "ok"}