from __future__ import annotations

from datetime import datetime

import requests
import streamlit as st

from src import config

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES_ID = "MORTGAGE30US"

REQUEST_TIMEOUT = 8  # seconds


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_latest_pmms_30yr() -> dict | None:
    """
    Fetch the latest Freddie Mac PMMS 30-year fixed mortgage rate
    using FRED's official REST API for the MORTGAGE30US series.

    Returns
    -------
    dict | None
        Example:
        {
            "rate": 6.36,
            "date": "2026-05-14",
            "date_formatted": "May 14, 2026",
            "source": "Freddie Mac PMMS via FRED"
        }
    """
    if not config.FRED_API_KEY:
        return None

    try:
        resp = requests.get(
            FRED_OBSERVATIONS_URL,
            params={
                "series_id": FRED_SERIES_ID,
                "api_key": config.FRED_API_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 5,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        for obs in data.get("observations", []):
            if obs.get("value") in (None, "."):
                continue

            parsed_date = datetime.strptime(obs["date"], "%Y-%m-%d")
            return {
                "rate": float(obs["value"]),
                "date": obs["date"],
                "date_formatted": f"{parsed_date:%B} {parsed_date.day}, {parsed_date.year}",
                "source": "Freddie Mac PMMS via FRED",
            }

        return None

    except Exception:
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
