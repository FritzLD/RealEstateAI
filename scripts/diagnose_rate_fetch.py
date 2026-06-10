"""
diagnose_rate_fetch.py
=======================
One-off diagnostic: calls the FRED PMMS endpoint directly, bypassing
Streamlit's cache and the try/except in rate_service.py, and prints
the full response or full exception so we can see exactly what is
failing on this machine.

Usage:
    python scripts/diagnose_rate_fetch.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from src.rate_service import FRED_PMMS_30YR_URL, REQUEST_TIMEOUT

print(f"requests version: {requests.__version__}")
print(f"URL: {FRED_PMMS_30YR_URL}")
print(f"Timeout: {REQUEST_TIMEOUT}s")
print("-" * 60)

try:
    resp = requests.get(FRED_PMMS_30YR_URL, timeout=REQUEST_TIMEOUT)
    print(f"Status code: {resp.status_code}")
    print(f"Response length: {len(resp.text)} chars")
    print("\nFirst 200 chars:")
    print(resp.text[:200])
    print("\nLast 200 chars:")
    print(resp.text[-200:])
except Exception:
    print("EXCEPTION:")
    traceback.print_exc()
