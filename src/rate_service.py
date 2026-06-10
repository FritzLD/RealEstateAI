from __future__ import annotations

import io

import pandas as pd
import requests
import streamlit as st


FRED_PMMS_30YR_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"

# pd.read_csv() on a URL has no timeout and can hang indefinitely if FRED
# doesn't respond. Fetch with requests + a short timeout so the chat never
# hangs waiting on this and falls back to stored rate data instead.
REQUEST_TIMEOUT = 8  # seconds


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_latest_pmms_30yr() -> dict | None:
    """
    Fetch the latest Freddie Mac PMMS 30-year fixed mortgage rate
    using the FRED MORTGAGE30US series.

    Returns
    -------
    dict | None
        Example:
        {
            "rate": 6.36,
            "date": "2026-05-14",
            "source": "Freddie Mac PMMS via FRED"
        }
    """
    try:
        resp = requests.get(
            FRED_PMMS_30YR_URL,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; RealEstateAI/1.0)"},
        )
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text))
        df = df.dropna()
        df = df[df["MORTGAGE30US"] != "."]

        latest = df.iloc[-1]

        return {
            "rate": float(latest["MORTGAGE30US"]),
            "date": str(latest["observation_date"]),
            "source": "Freddie Mac PMMS via FRED",
        }

    except Exception:
        return None


def build_pmms_context() -> str:
    """
    Build a clean text block that can be passed into the RAG chain.
    """
    pmms = get_latest_pmms_30yr()

    if not pmms:
        return (
            "Latest Freddie Mac PMMS rate could not be retrieved. "
            "Do not provide a current mortgage rate unless it is available in the retrieved context, "
            "and do not imply that any stored rate is today's live pricing."
        )

    return f"""
Latest Freddie Mac PMMS benchmark:
- 30-year fixed-rate mortgage survey rate: {pmms["rate"]:.2f}%
- Survey date: {pmms["date"]}
- Source: {pmms["source"]}
- This is a national survey benchmark only.
- For a actual interest rate in Ohio, Kentucky or Florida please contact Frederick Duff at (502)345-0682 or you may apply online for a free evaluation at www.pre-qualifymymortgage.com. Queen City is a Equal Opportunity Lender. 
"""
