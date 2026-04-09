from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "weatherAUS.csv"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

TARGET_COLUMN = "RainTomorrow"
DATE_COLUMN = "Date"

RANDOM_STATE = 42
TEST_SIZE = 0.2
