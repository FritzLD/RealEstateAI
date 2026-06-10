"""
diagnose_rate_fetch.py
=======================
One-off diagnostic: calls get_latest_pmms_30yr() and build_pmms_context()
directly (bypassing Streamlit's UI) to confirm the FRED API key is set
and the live PMMS rate can be fetched.

Usage:
    python scripts/diagnose_rate_fetch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.rate_service import build_pmms_context, get_latest_pmms_30yr

print(f"FRED_API_KEY set: {bool(config.FRED_API_KEY)}")
print("-" * 60)

result = get_latest_pmms_30yr()
print(f"get_latest_pmms_30yr() -> {result}")
print("-" * 60)
print(build_pmms_context())
