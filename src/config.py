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

# ── SARIMAX fixed orders ───────────────────────────────────────────────────────
# Standard orders for monthly real estate data with annual seasonality.
# Using fixed orders eliminates the slow auto_arima grid search on every load,
# reducing forecast fitting from ~90 seconds to ~15 seconds.
# (p,d,q) non-seasonal:  AR=1, one difference, MA=1
# (P,D,Q,m) seasonal:    AR=1, one seasonal difference, MA=1, period=12 months
SARIMAX_ORDER          = (1, 1, 1)
SARIMAX_SEASONAL_ORDER = (1, 1, 1, 12)

# ── Refinancing analysis defaults ─────────────────────────────────────────────
REFI_THRESHOLD_PP   = 0.75   # pp above current rate to flag refi window
REFI_LOAN_AMOUNTS   = [150_000, 300_000]  # $ loan sizes for savings calc

# ── LLM / embeddings ─────────────────────────────────────────────────────────
# Read from .env locally; fall back to Streamlit secrets when deployed
def _get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            pass
    return key

OPENAI_API_KEY   = _get_api_key()
DEFAULT_MODEL    = "gpt-4o-mini"
EMBEDDING_MODEL  = "text-embedding-3-small"
TOP_K_RETRIEVAL  = 6

# ── Streamlit ─────────────────────────────────────────────────────────────────
APP_TITLE = "RealEstateAI – Dayton MSA Intelligence"
APP_ICON  = "🏠"
