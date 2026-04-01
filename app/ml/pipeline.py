import pandas as pd
import mlflow

from autogluon.tabular import TabularPredictor

from app.core.config import DATA_PATH, MODEL_DIR, LEADERBOARD_PATH
from app.core.logger import logger
from app.core.exceptions import CustomException


def run_pipeline(target: str):

    try:
        logger.info("AutoGluon training started")

        df = pd.read_csv(DATA_PATH)

        # Train-test split
        train_data = df.sample(frac=0.8, random_state=42)
        test_data = df.drop(train_data.index)

        with mlflow.start_run():

            predictor = TabularPredictor(
                label=target,
                path=MODEL_DIR
            ).fit(
                train_data,
                time_limit=60,
                presets="medium_quality"
            )

            performance = predictor.evaluate(test_data)

            leaderboard = predictor.leaderboard(test_data)
            leaderboard.to_csv(LEADERBOARD_PATH, index=False)

            mlflow.log_metric("score", list(performance.values())[0])

        logger.info("Training completed")

        return {
            "performance": performance,
            "best_model": predictor.get_model_best()
        }

    except Exception as e:
        logger.error(str(e))
        raise CustomException(str(e))