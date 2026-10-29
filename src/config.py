"""Configurazione centrale del progetto."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

RAW_CSV = DATA_DIR / "tickets_raw.csv"
TRAIN_FILE = DATA_DIR / "train.parquet"
TEST_FILE = DATA_DIR / "test.parquet"
CLASSIFIER_FILE = MODELS_DIR / "queue_classifier.joblib"
INDEX_EMB_FILE = MODELS_DIR / "kb_embeddings.npy"
INDEX_META_FILE = MODELS_DIR / "kb_meta.parquet"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# OpenAI
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")

# Guardrail
CONFIDENCE_THRESHOLD = 0.70   # sotto questa probabilita' il ticket va in revisione umana
SIMILARITY_THRESHOLD = 0.45   # sotto questa similarita' coseno non si genera una bozza
TOP_K = 5

for d in (DATA_DIR, MODELS_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
