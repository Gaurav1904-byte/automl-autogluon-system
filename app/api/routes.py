from fastapi import APIRouter, UploadFile
import pandas as pd

from app.core.config import DATA_PATH
from app.ml.pipeline import run_pipeline
from app.services.predictor import predict_single

router = APIRouter()


@router.post("/upload")
async def upload(file: UploadFile):
    df = pd.read_csv(file.file)
    df.to_csv(DATA_PATH, index=False)
    return {"message": "Dataset uploaded"}


@router.post("/train")
def train(target: str):
    return run_pipeline(target)


@router.post("/predict")
def predict(data: dict):
    return {"prediction": predict_single(data)}