"""Central configuration for RealEstateAI."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root and load .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Data files ────────────────────────────────────────────────────────────────
# Data lives in the project's data/ folder (works locally and on Streamlit Cloud)
DATA_DIR = PROJECT_ROOT / "data"

STATS_CSV      = DATA_DIR / "statisticsday.csv"
RATES_XLSX     = DATA_DIR / "historicalweeklydata.xlsx"
INFLATION_XLSX = DATA_DIR / "inflation.xlsx"

# ChromaDB persistence
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# ── Forecasting constants ─────────────────────────────────────────────────────
FORECAST_HORIZON = 12      # months to forecast forward
SEQ_LENGTH       = 12      # lookback window for sequence models
MA_SHORT         = 3       # short moving-average window (months)
MA_LONG          = 12      # long moving-average window (months)
VIF_THRESHOLD    = 10.0    # drop exog columns above this VIF
TEST_HOLDOUT     = 12      # months held out for model evaluation

# ── Refinancing analysis defaults ─────────────────────────────────────────────
REFI_THRESHOLD_PP   = 0.75   # pp above current rate to flag refi window
REFI_LOAN_AMOUNTS   = [150_000, 300_000]  # $ loan sizes for savings calc

# ── LLM / embeddings ─────────────────────────────────────────────────────────
# Read from .env locally; fall back to Streamlit secrets when deployed
def _get_secret(env_var: str) -> str:
    value = os.getenv(env_var, "")
    if not value:
        try:
            import streamlit as st
            value = st.secrets.get(env_var, "")
        except Exception:
            pass
    return value

OPENAI_API_KEY   = _get_secret("OPENAI_API_KEY")
DEFAULT_MODEL    = "gpt-4o-mini"
EMBEDDING_MODEL  = "text-embedding-3-small"
TOP_K_RETRIEVAL  = 6

# Free key from https://fred.stlouisfed.org/docs/api/api_key.html
# Used to fetch the live Freddie Mac PMMS 30-year mortgage rate.
FRED_API_KEY = _get_secret("FRED_API_KEY")

# ── Streamlit ─────────────────────────────────────────────────────────────────
APP_TITLE = "RealEstateAI – Dayton MSA Intelligence"
APP_ICON  = "🏠"
