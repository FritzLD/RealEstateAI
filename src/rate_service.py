from __future__ import annotations

import csv
import io
import logging
from datetime import datetime

import requests
import streamlit as st

from src import config

logger = logging.getLogger(__name__)

# Primary source: FRED's official REST API (needs FRED_API_KEY in secrets / .env)
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES_ID = "MORTGAGE30US"

# Fallback source: Freddie Mac's own public PMMS history file (no key needed)
FREDDIE_HISTORY_CSV_URL = "https://www.freddiemac.com/pmms/docs/PMMS_history.csv"

REQUEST_TIMEOUT = 10  # seconds
HEADERS = {"User-Agent": "RealEstateAI/1.0 (Ohio Valley Analytics)"}

# Reason the most recent fetch failed (for logs and scripts/diagnose_rate_fetch.py)
LAST_ERROR: str | None = None


def _format(rate: float, date_str: str, source: str) -> dict:
    parsed = datetime.strptime(date_str, "%Y-%m-%d")
    return {
        "rate": rate,
        "date": date_str,
        "date_formatted": f"{parsed:%B} {parsed.day}, {parsed.year}",
        "source": source,
    }


def _fetch_from_fred() -> dict:
    """Latest MORTGAGE30US observation from FRED. Raises on any failure."""
    if not config.FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY is not set (check Streamlit secrets or .env)")

    resp = requests.get(
        FRED_OBSERVATIONS_URL,
        params={
            "series_id": FRED_SERIES_ID,
            "api_key": config.FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 5,
        },
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code != 200:
        # FRED returns a JSON error_message (e.g. bad API key) — surface it
        try:
            detail = resp.json().get("error_message", resp.text[:200])
        except ValueError:
            detail = resp.text[:200]
        raise RuntimeError(f"FRED HTTP {resp.status_code}: {detail}")

    for obs in resp.json().get("observations", []):
        if obs.get("value") not in (None, "", "."):
            return _format(float(obs["value"]), obs["date"], "Freddie Mac PMMS via FRED")

    raise RuntimeError("FRED returned no usable MORTGAGE30US observations")


def _parse_date(raw: str) -> str:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"unrecognized date {raw!r}")


def _fetch_from_freddie_csv() -> dict:
    """Latest 30-yr rate from Freddie Mac's PMMS history CSV. Raises on any failure."""
    resp = requests.get(FREDDIE_HISTORY_CSV_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"Freddie Mac CSV HTTP {resp.status_code}")

    reader = csv.DictReader(io.StringIO(resp.text.lstrip("﻿")))
    fields = {f.strip().lower(): f for f in (reader.fieldnames or [])}
    date_col = fields.get("date")
    rate_col = fields.get("pmms30")
    if not date_col or not rate_col:
        raise RuntimeError(f"Freddie Mac CSV columns changed: {reader.fieldnames}")

    latest: tuple[str, float] | None = None
    for row in reader:
        raw_rate = (row.get(rate_col) or "").strip()
        raw_date = (row.get(date_col) or "").strip()
        if not raw_rate or not raw_date:
            continue
        try:
            item = (_parse_date(raw_date), float(raw_rate))
        except ValueError:
            continue
        if latest is None or item[0] > latest[0]:
            latest = item

    if latest is None:
        raise RuntimeError("Freddie Mac CSV contained no usable 30-year rows")
    return _format(latest[1], latest[0], "Freddie Mac PMMS")


@st.cache_data(ttl=60 * 60 * 3, show_spinner=False)
def _fetch_latest_cached() -> dict:
    """
    Try FRED first, then Freddie Mac's CSV. Raises if both fail.

    Raising (instead of returning None) matters: st.cache_data does not cache
    exceptions, so a temporary outage is retried on the next question rather
    than being remembered for hours.
    """
    errors = []
    for fetch in (_fetch_from_fred, _fetch_from_freddie_csv):
        try:
            return fetch()
        except Exception as exc:  # noqa: BLE001 — record every reason
            errors.append(f"{fetch.__name__}: {exc}")
    raise RuntimeError(" | ".join(errors))


def get_latest_pmms_30yr() -> dict | None:
    """
    Latest Freddie Mac PMMS 30-year fixed rate, or None if no source responds.

    Example return:
        {"rate": 6.36, "date": "2026-05-14", "date_formatted": "May 14, 2026",
         "source": "Freddie Mac PMMS via FRED"}
    """
    global LAST_ERROR
    try:
        result = _fetch_latest_cached()
        LAST_ERROR = None
        return result
    except Exception as exc:  # noqa: BLE001
        LAST_ERROR = str(exc)
        logger.warning("Live PMMS rate unavailable: %s", LAST_ERROR)
        print(f"[rate_service] Live PMMS rate unavailable: {LAST_ERROR}", flush=True)
        return None


def build_pmms_context() -> str:
    """
    Build a clean text block that can be passed into the RAG chain.
    """
    pmms = get_latest_pmms_30yr()

    if not pmms:
        return (
            "Today's live Freddie Mac PMMS rate could not be retrieved. "
            "Do not provide a current mortgage rate unless it is available in the retrieved context, "
            "and do not imply that any stored rate is today's live pricing."
        )

    return f"""
Latest Freddie Mac PMMS benchmark:
- 30-year fixed-rate mortgage survey rate: {pmms["rate"]:.2f}% as of {pmms["date_formatted"]}
- Source: {pmms["source"]}
- PMMS results are released weekly on Thursdays at 12 p.m. ET and are an average of loan
  rates offered the prior Thursday through Wednesday. Always state the rate together with
  this "as of {pmms["date_formatted"]}" date — never describe it as today's rate.
- This is a national survey benchmark only, not Fred's pricing and not a live rate quote.
- For a actual interest rate in Ohio, Kentucky or Florida please contact Frederick Duff at (502)345-0682 or you may apply online for a free evaluation at www.pre-qualifymymortgage.com. Queen City is a Equal Opportunity Lender.
"""
