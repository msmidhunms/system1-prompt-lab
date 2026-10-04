"""Configuration for the backend."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = PROJECT_ROOT / "backend" / "data.db"
MODELS_CACHE_DIR = PROJECT_ROOT / "models_cache"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_CACHE_DIR.mkdir(exist_ok=True)

# Environment settings
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
TOP_N_FEEDBACK = int(os.getenv("TOP_N_FEEDBACK", "5"))

# Model settings
LAYA_MODEL_NAME = os.getenv("LAYA_MODEL_NAME", "layalm/laya1")  # HuggingFace model name
LAYA_DEVICE = os.getenv("LAYA_DEVICE", "cpu")  # or "cuda"

# Database
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Frontend
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# Eval
EVAL_HOLDOUT_RATIO = float(os.getenv("EVAL_HOLDOUT_RATIO", "0.1"))  # 10% holdout
EVAL_DEV_RATIO = float(os.getenv("EVAL_DEV_RATIO", "0.9"))  # 90% for loop to optimize
