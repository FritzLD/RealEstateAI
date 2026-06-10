"""
diagnose_rate_fetch.py
=======================
One-off diagnostic: tries the FRED PMMS CSV endpoint plus a few related
URLs, bypassing Streamlit's cache and the try/except in rate_service.py,
so we can see exactly which hosts/paths succeed or hang on this machine.

Usage:
    python scripts/diagnose_rate_fetch.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from src.rate_service import FRED_PMMS_30YR_URL, REQUEST_TIMEOUT

CHECKS = [
    ("FRED graph CSV (current code path)", FRED_PMMS_30YR_URL, REQUEST_TIMEOUT),
    ("FRED graph CSV (longer timeout)", FRED_PMMS_30YR_URL, 25),
    ("FRED series page (works in browser)", "https://fred.stlouisfed.org/series/MORTGAGE30US", REQUEST_TIMEOUT),
    ("FRED API host (different subdomain)", "https://api.stlouisfed.org/fred/series/observations?series_id=MORTGAGE30US&file_type=json", REQUEST_TIMEOUT),
    ("Sanity check (google.com)", "https://www.google.com", REQUEST_TIMEOUT),
]

print(f"requests version: {requests.__version__}")
print("=" * 60)

for label, url, timeout in CHECKS:
    print(f"\n{label}")
    print(f"  URL: {url}")
    print(f"  Timeout: {timeout}s")
    start = time.monotonic()
    try:
        resp = requests.get(url, timeout=timeout)
        elapsed = time.monotonic() - start
        print(f"  -> Status {resp.status_code} in {elapsed:.2f}s, {len(resp.content)} bytes")
    except Exception as exc:
        elapsed = time.monotonic() - start
        print(f"  -> FAILED after {elapsed:.2f}s: {type(exc).__name__}: {exc}")

print("\n" + "=" * 60)
print("Done.")
