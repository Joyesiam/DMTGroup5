"""
Central configuration for Assignment 1 AutoResearch.
All paths, constants, and hyperparameter defaults in one place.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ITERATIONS_DIR = BASE_DIR / "iterations"
DOCS_DIR = BASE_DIR / "docs"

# Source data (from the original assignment folder)
SOURCE_DATA_DIR = BASE_DIR.parent / "Assignment 1 (Advanced)" / "data"
RAW_DATA_FILE = SOURCE_DATA_DIR / "dataset_mood_smartphone.csv"

# Reproducibility
RANDOM_SEED = 42

# MacBook constraint: never use verbose output, limit parallelism
VERBOSE = 0
N_JOBS = 1  # NEVER use -1: spawns 14 workers, each copies all data, crashes Mac
MAX_MEMORY_MB = 8000  # Kill process if it exceeds this

# Dataset structure
ID_COL = "id"
DATE_COL = "date"
TARGET_COL = "target_mood"

# Variable categories (from original utils.py)
MOOD_VARS = ["mood"]
SELF_REPORT_VARS = ["circumplex.arousal", "circumplex.valence"]
SENSOR_VARS = ["activity", "screen", "call", "sms"]
# APP_VARS = [
#     "appCat.builtin", "appCat.communication", "appCat.entertainment",
#     "appCat.finance", "appCat.game", "appCat.office", "appCat.other",
#     "appCat.social", "appCat.travel", "appCat.unknown",
#     "appCat.utilities", "appCat.weather",
# ]
APP_VARS = [
    "appCat.builtin", "appCat.communication", "appCat.entertainment",
    "appCat.finance", "appCat.game", "appCat.office", "appCat.other",
    "appCat.social", "appCat.travel",
    "appCat.utilities", "appCat.weather",
]
ALL_VARS = MOOD_VARS + SELF_REPORT_VARS + SENSOR_VARS + APP_VARS

# Aggregation rules per variable type
MEAN_VARS = MOOD_VARS + SELF_REPORT_VARS + ["activity"]
SUM_VARS = ["screen"] + APP_VARS
COUNT_VARS = ["call", "sms"]

# Feature engineering defaults
DEFAULT_WINDOW_SIZE = 7
N_LAGS = 3

# Classification
N_CLASSES = 3
CLASS_LABELS = ["Low", "Medium", "High"]

# Train/test split
TEST_FRACTION = 0.2

# Cross-validation
N_CV_FOLDS = 5

# Ensure output dirs exist
DOCS_DIR.mkdir(parents=True, exist_ok=True)
ITERATIONS_DIR.mkdir(parents=True, exist_ok=True)
