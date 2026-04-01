from autogluon.tabular import TabularPredictor
import pandas as pd

from app.core.config import MODEL_DIR


def predict_single(data: dict):

    predictor = TabularPredictor.load(MODEL_DIR)

    df = pd.DataFrame([data])

    prediction = predictor.predict(df)

    return prediction.tolist()