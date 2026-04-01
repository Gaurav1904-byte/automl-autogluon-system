from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="AutoML AutoGluon System")

app.include_router(router)
