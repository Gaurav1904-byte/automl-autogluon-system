import os

ARTIFACT_DIR = "artifacts"
MODEL_DIR = os.path.join(ARTIFACT_DIR, "autogluon_model")
DATA_PATH = os.path.join(ARTIFACT_DIR, "data.csv")
LEADERBOARD_PATH = os.path.join(ARTIFACT_DIR, "leaderboard.csv")