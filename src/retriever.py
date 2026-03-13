"""
RealEstateRetriever
Hybrid retriever: vector similarity search + keyword-triggered pandas stat lookups.

Architecture mirrors InsightForge's CustomBusinessRetriever.
Real-estate-specific keyword sets trigger stat helpers:
  • forecast   → 12-month SARIMAX outlook
  • inventory  → market disparity / Active-Sales gap
  • rates/refi → current mortgage rate + refi windows
  • trends     → YoY and 6-month directional changes
  • month name → year-over-year breakdown for that specific month
"""

from __future__ import annotations

import re
from typing import Any, List

import pandas as pd
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.data_loader import RealEstateDataLoader
from src.refi_analysis import RefiAnalyzer


_FORECAST_KEYWORDS  = {"forecast", "predict", "projection", "outlook", "future",
                        "expect", "next", "upcoming", "months"}
_INVENTORY_KEYWORDS = {"inventory", "disparity", "listing", "listings", "active",
                        "supply", "demand", "market condition", "buyer", "seller"}
_RATE_KEYWORDS      = {"rate", "rates", "mortgage", "30 year", "30yr", "interest",
                        "refinanc", "refi", "payment", "afford"}
_TREND_KEYWORDS     = {"trend", "change", "growing", "declining", "increase",
                        "decrease", "year over year", "yoy", "momentum", "direction"}

# Map month names (and abbreviations) to month number
_MONTH_MAP = {
    "january": 1,  "jan": 1,
    "february": 2, "feb": 2,
    "march": 3,    "mar": 3,
    "april": 4,    "apr": 4,
    "may": 5,
    "june": 6,     "jun": 6,
    "july": 7,     "jul": 7,
    "august": 8,   "aug": 8,
    "september": 9,"sep": 9,  "sept": 9,
    "october": 10, "oct": 10,
    "november": 11,"nov": 11,
    "december": 12,"dec": 12,
}


def _tokenize(text: str) -> set[str]:
    return set(text.lower().split())


def _detect_month(query: str) -> int | None:
    """Return month number if a month name appears in the query, else None."""
    q = query.lower()
    for name, num in _MONTH_MAP.items():
        # word-boundary match so "march" doesn't match inside another word
        if re.search(rf"\b{name}\b", q):
            return num
    return None


def _detect_years_requested(query: str) -> int:
    """Extract how many years back the user wants (e.g. 'last 10 years' → 10)."""
    m = re.search(r"last\s+(\d+)\s+year", query.lower())
    if m:
        return int(m.group(1))
    return 10   # default to 10 if not specified


class RealEstateRetriever(BaseRetriever):
    """
    Hybrid retriever combining vector search with on-demand stat docs.

    Attributes must be declared at class level for Pydantic v2 / BaseRetriever.
    """

    vector_retriever: Any
    loader: Any         # RealEstateDataLoader
    forecaster: Any | None = None   # MarketForecaster (optional, lazy)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # 1. Vector similarity search
        docs: List[Document] = self.vector_retriever.invoke(query)

        # 2. Keyword augmentation
        tokens = _tokenize(query)

        if tokens & _FORECAST_KEYWORDS:
            doc = self._forecast_doc()
            if doc:
                docs.append(doc)

        if tokens & _INVENTORY_KEYWORDS:
            doc = self._disparity_doc()
            if doc:
                docs.append(doc)

        if tokens & _RATE_KEYWORDS:
            doc = self._rate_doc()
            if doc:
                docs.append(doc)

        if tokens & _TREND_KEYWORDS:
            doc = self._trend_doc()
            if doc:
                docs.append(doc)

        # 3. Month-specific year-over-year lookup
        month_num = _detect_month(query)
        if month_num is not None:
            n_years = _detect_years_requested(query)
            doc = self._monthly_yoy_doc(month_num, n_years)
            if doc:
                docs.append(doc)

        # 4. Deduplicate by leading 80 chars, cap at 9
        seen: set[str] = set()
        unique: List[Document] = []
        for d in docs:
            key = d.page_content[:80]
            if key not in seen:
                seen.add(key)
                unique.append(d)
            if len(unique) >= 9:
                break

        return unique

    # ── Stat helpers ──────────────────────────────────────────────────────────

    def _forecast_doc(self) -> Document | None:
        if self.forecaster is None:
            return None
        try:
            summary = self.forecaster.get_forecast_summary()
            return Document(page_content=summary, metadata={"type": "live_forecast"})
        except Exception:
            return None

    def _disparity_doc(self) -> Document | None:
        try:
            d = self.loader.get_disparity_stats()
            content = (
                f"Live Inventory-Sales Disparity Data:\n"
                f"- Current disparity: {d['current_disparity']:,} units\n"
                f"- Current disparity %: {d['current_disparity_pct']}% "
                f"({d['market_condition']})\n"
                f"- 12-month average disparity %: {d['avg_disparity_pct_12mo']}%\n"
                f"- Balanced market threshold: {d['balanced_threshold_pct']}%"
            )
            return Document(page_content=content, metadata={"type": "live_disparity"})
        except Exception:
            return None

    def _rate_doc(self) -> Document | None:
        try:
            r = self.loader.get_rate_stats()
            if not r:
                return None
            refi = RefiAnalyzer(self.loader.df)
            refi_summary = refi.generate_refi_summary()
            content = (
                f"Current Mortgage Rate Data:\n"
                f"- Current 30-year fixed rate: {r['current_rate']}%\n"
                f"- Rate 1 year ago: {r['rate_1yr_ago']}%\n"
                f"- 5-year average: {r['rate_5yr_avg']}%\n\n"
                f"{refi_summary}"
            )
            return Document(page_content=content, metadata={"type": "live_rates"})
        except Exception:
            return None

    def _trend_doc(self) -> Document | None:
        try:
            t = self.loader.get_trend_stats()
            content = (
                f"Recent Market Trend Data:\n"
                f"- Sales trend (last 6 mo vs prior 6): {t['sales_trend_6mo']:+}%\n"
                f"- Active listings trend (last 6 mo vs prior 6): {t['active_trend_6mo']:+}%\n"
                f"- Peak historical sales: {t['peak_sales_value']:,} units in {t['peak_sales_month']}\n"
                f"- Historical sales low: {t['low_sales_value']:,} units in {t['low_sales_month']}"
            )
            return Document(page_content=content, metadata={"type": "live_trends"})
        except Exception:
            return None

    def _monthly_yoy_doc(self, month_num: int, n_years: int = 10) -> Document | None:
        """
        Build a year-over-year table for a specific calendar month.
        E.g. month_num=3 → March data for each of the last n_years years.
        """
        try:
            df   = self.loader.df
            month_name = list(_MONTH_MAP.keys())[
                [v for v in _MONTH_MAP.values()].index(month_num)
            ].capitalize()

            # Filter to just the requested month
            month_df = df[df.index.month == month_num].copy()
            if month_df.empty:
                return None

            # Limit to last n_years
            latest_year = month_df.index.year.max()
            cutoff_year = latest_year - n_years + 1
            month_df    = month_df[month_df.index.year >= cutoff_year]

            # Build rows
            rows = []
            prev_active = None
            prev_sales  = None
            for _, row in month_df.sort_index().iterrows():
                yr     = row.name.year
                active = int(row["Active"]) if pd.notna(row.get("Active")) else None
                sales  = int(row["Sales"])  if pd.notna(row.get("Sales"))  else None

                active_chg = ""
                sales_chg  = ""
                if prev_active and active:
                    pct = (active - prev_active) / prev_active * 100
                    active_chg = f" ({pct:+.1f}% YoY)"
                if prev_sales and sales:
                    pct = (sales - prev_sales) / prev_sales * 100
                    sales_chg = f" ({pct:+.1f}% YoY)"

                rate_str = ""
                if "30yrFRM" in row.index and pd.notna(row["30yrFRM"]):
                    rate_str = f", rate {row['30yrFRM']:.2f}%"

                rows.append(
                    f"  {yr}: Active listings {active:,}{active_chg}, "
                    f"Sales {sales:,}{sales_chg}{rate_str}"
                )
                prev_active = active
                prev_sales  = sales

            content = (
                f"Dayton MSA – {month_name} Year-over-Year Data "
                f"(last {n_years} years, {cutoff_year}–{latest_year}):\n"
                + "\n".join(rows)
            )
            return Document(page_content=content, metadata={"type": "monthly_yoy"})
        except Exception:
            return None
