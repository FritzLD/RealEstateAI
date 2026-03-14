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
_PRICE_KEYWORDS         = {"price", "sale price", "home price", "average price", "avg price",
                           "dollar", "volume", "sales volume", "per home", "cost", "value",
                           "how much", "median", "expensive", "affordable"}
_SUPPLY_KEYWORDS        = {"absorption", "mos", "turnover", "months supply", "sellout",
                           "sell all", "how long"}
_AFFORDABILITY_KEYWORDS = {"payment", "afford", "qualify", "income", "borrow", "piti",
                           "down", "loan", "principal", "interest", "calculator"}
_APPRECIATION_KEYWORDS  = {"appreciation", "appreciate", "cagr", "compound", "equity",
                           "growth rate", "invest", "return", "built"}
_SEASONAL_PRICE_KEYWORDS = {"season", "seasonal", "spring", "summer", "fall", "winter",
                             "quarter", "quarterly", "best time", "time of year"}

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

        if tokens & _PRICE_KEYWORDS:
            doc = self._price_doc()
            if doc:
                docs.append(doc)

        if tokens & _SUPPLY_KEYWORDS:
            doc = self._supply_doc()
            if doc:
                docs.append(doc)

        if tokens & _AFFORDABILITY_KEYWORDS:
            doc = self._affordability_doc()
            if doc:
                docs.append(doc)

        if tokens & _APPRECIATION_KEYWORDS:
            doc = self._appreciation_doc()
            if doc:
                docs.append(doc)

        if tokens & _SEASONAL_PRICE_KEYWORDS:
            doc = self._seasonal_price_doc()
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
            if len(unique) >= 14:
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

    def _price_doc(self) -> Document | None:
        """
        Compute average sale price per home by year (Sales_Volume / Sales)
        and return as a formatted table for the LLM context.
        """
        try:
            df = self.loader.df
            if "Sales_Volume" not in df.columns or "Sales" not in df.columns:
                return None

            annual = df.groupby(df.index.year).agg(
                total_volume=("Sales_Volume", "sum"),
                total_sales=("Sales",         "sum"),
            )
            annual = annual[annual["total_sales"] > 0]
            annual["avg_price"] = (annual["total_volume"] / annual["total_sales"]).round(0)

            lines = [
                "Dayton MSA – Average Sale Price Per Home by Year",
                "(calculated as: Annual Sales Volume / Annual Number of Sales)",
                "-" * 60,
            ]
            prev_price = None
            for yr, row in annual.iterrows():
                chg_str = ""
                if prev_price and prev_price > 0:
                    chg = (row["avg_price"] - prev_price) / prev_price * 100
                    chg_str = f"  ({chg:+.1f}% YoY)"
                lines.append(
                    f"  {yr}: ${row['avg_price']:,.0f}{chg_str}"
                    f"  |  {int(row['total_sales']):,} sales"
                    f"  |  ${row['total_volume']:,.0f} total volume"
                )
                prev_price = row["avg_price"]

            # Trailing 12-month figure
            recent  = df.tail(12)
            total_s = recent["Sales"].sum()
            if total_s > 0:
                recent_price = recent["Sales_Volume"].sum() / total_s
                lines.append(f"\nTrailing 12-Month Average Sale Price: ${recent_price:,.0f}/home")

            return Document(
                page_content="\n".join(lines),
                metadata={"type": "price_analysis"},
            )
        except Exception:
            return None

    def _supply_doc(self) -> Document | None:
        """
        Months of Supply (Active ÷ Sales) and Absorption Rate (Sales ÷ New_Listings)
        by year, plus current figures.
        """
        try:
            df = self.loader.df
            if "Active" not in df.columns or "Sales" not in df.columns:
                return None

            annual = df.groupby(df.index.year).agg(
                avg_active=("Active", "mean"),
                avg_sales=("Sales",  "mean"),
            )
            annual = annual[annual["avg_sales"] > 0]
            annual["mos"] = (annual["avg_active"] / annual["avg_sales"]).round(1)

            has_new = "New_Listings" in df.columns
            if has_new:
                ann_abs = df.groupby(df.index.year).agg(
                    total_sales=("Sales",        "sum"),
                    total_new  =("New_Listings", "sum"),
                )
                ann_abs = ann_abs[ann_abs["total_new"] > 0]
                ann_abs["abs_rate"] = (ann_abs["total_sales"] / ann_abs["total_new"] * 100).round(1)

            def _mos_label(m: float) -> str:
                if m < 2:   return "Very Hot Seller's Market"
                if m < 4:   return "Seller's Market"
                if m <= 6:  return "Balanced Market"
                if m <= 9:  return "Buyer's Market"
                return "Very Slow Buyer's Market"

            lines = [
                "Dayton MSA – Months of Supply & Absorption Rate by Year",
                "(Months of Supply = Active ÷ Monthly Sales  |  <2=Very Hot  2-4=Seller's  4-6=Balanced  >6=Buyer's)",
            ]
            if has_new:
                lines.append("(Absorption Rate = Annual Sales ÷ New Listings × 100  |  >80%=Very Hot  60-80%=Active  <40%=Slow)")
            lines.append("-" * 65)

            for yr, row in annual.iterrows():
                abs_str = ""
                if has_new and yr in ann_abs.index:
                    abs_str = f"  |  abs. rate {ann_abs.loc[yr,'abs_rate']:.0f}%"
                lines.append(
                    f"  {yr}: {row['mos']:.1f} mos supply  ({_mos_label(row['mos'])}){abs_str}"
                )

            # Current month
            latest = df.iloc[-1]
            if latest["Sales"] > 0:
                cur_mos = latest["Active"] / latest["Sales"]
                lines.append(f"\nCurrent Month: {cur_mos:.1f} months of supply  ({_mos_label(cur_mos)})")

            return Document(page_content="\n".join(lines), metadata={"type": "supply_analysis"})
        except Exception:
            return None

    def _affordability_doc(self) -> Document | None:
        """
        Monthly P&I payment scenarios at current avg sale price for 3.5%, 5%, 10%, 20% down.
        Also estimates the annual income needed using the 28% front-end ratio.
        """
        try:
            df = self.loader.df
            if "Sales_Volume" not in df.columns or "Sales" not in df.columns:
                return None
            if "30yrFRM" not in df.columns:
                return None

            # Trailing 12-month avg sale price
            recent  = df.tail(12)
            total_s = recent["Sales"].sum()
            if total_s == 0:
                return None
            avg_price    = recent["Sales_Volume"].sum() / total_s
            current_rate = float(df["30yrFRM"].dropna().iloc[-1])
            monthly_r    = current_rate / 100 / 12
            n            = 360  # 30-year fixed

            def pni(price: float, down_pct: float) -> float:
                loan = price * (1 - down_pct)
                if monthly_r == 0:
                    return loan / n
                return loan * (monthly_r * (1 + monthly_r) ** n) / ((1 + monthly_r) ** n - 1)

            lines = [
                "Dayton MSA – Affordability & Monthly Payment Scenarios",
                f"Trailing 12-Month Average Sale Price: ${avg_price:,.0f}",
                f"Current 30-Year Fixed Rate: {current_rate:.2f}%",
                "(P&I = Principal & Interest only — does not include taxes, insurance, or HOA)",
                "-" * 65,
            ]
            for down_pct, label in [
                (0.035, "3.5% down (FHA minimum)"),
                (0.050, "5% down"),
                (0.100, "10% down"),
                (0.200, "20% down (no PMI)"),
            ]:
                pmt      = pni(avg_price, down_pct)
                loan     = avg_price * (1 - down_pct)
                down_amt = avg_price * down_pct
                inc_28   = pmt / 0.28 * 12   # gross income needed at 28% front-end ratio
                lines.append(
                    f"  {label}:"
                    f"  ${down_amt:,.0f} down"
                    f"  |  ${loan:,.0f} loan"
                    f"  |  ${pmt:,.0f}/mo P&I"
                    f"  |  ~${inc_28:,.0f}/yr income (28% rule)"
                )

            # Rate sensitivity: same price, 20% down, at ±1% and ±2% rates
            lines.append(f"\nRate Sensitivity (avg price, 20% down, 30-yr fixed):")
            for delta in [-2, -1, 0, +1, +2]:
                r = current_rate + delta
                if r <= 0:
                    continue
                mr = r / 100 / 12
                loan = avg_price * 0.80
                pmt = loan * (mr * (1 + mr) ** n) / ((1 + mr) ** n - 1)
                marker = " ← current" if delta == 0 else ""
                lines.append(f"  {r:.2f}%:  ${pmt:,.0f}/mo P&I{marker}")

            return Document(page_content="\n".join(lines), metadata={"type": "affordability"})
        except Exception:
            return None

    def _appreciation_doc(self) -> Document | None:
        """
        Home price appreciation: year-over-year changes, CAGR over full history,
        plus 5-year and 10-year CAGRs.
        """
        try:
            df = self.loader.df
            if "Sales_Volume" not in df.columns or "Sales" not in df.columns:
                return None

            annual = df.groupby(df.index.year).agg(
                total_volume=("Sales_Volume", "sum"),
                total_sales =("Sales",         "sum"),
            )
            annual = annual[annual["total_sales"] > 0]
            annual["avg_price"] = annual["total_volume"] / annual["total_sales"]

            if len(annual) < 2:
                return None

            first_yr    = int(annual.index[0])
            last_yr     = int(annual.index[-1])
            first_price = float(annual.iloc[0]["avg_price"])
            last_price  = float(annual.iloc[-1]["avg_price"])
            n_yrs       = last_yr - first_yr

            cagr_full  = ((last_price / first_price) ** (1 / n_yrs) - 1) * 100 if n_yrs > 0 else 0
            total_appr = (last_price / first_price - 1) * 100

            lines = [
                "Dayton MSA – Home Price Appreciation Analysis",
                "-" * 60,
                f"  Earliest year ({first_yr}): ${first_price:,.0f} avg sale price",
                f"  Latest year ({last_yr}):   ${last_price:,.0f} avg sale price",
                f"  Total appreciation ({first_yr}–{last_yr}): {total_appr:+.1f}%",
                f"  Full-period CAGR: {cagr_full:.2f}%/year",
                "",
                "Year-over-Year Price Changes:",
            ]

            prev_price = None
            for yr, row in annual.iterrows():
                if prev_price and prev_price > 0:
                    yoy = (row["avg_price"] - prev_price) / prev_price * 100
                    lines.append(f"  {yr}: ${row['avg_price']:,.0f}  ({yoy:+.1f}% YoY)")
                else:
                    lines.append(f"  {yr}: ${row['avg_price']:,.0f}")
                prev_price = row["avg_price"]

            # 5-year and 10-year CAGRs
            for lookback in [5, 10]:
                target = last_yr - lookback
                if target in annual.index:
                    s_price = float(annual.loc[target, "avg_price"])
                    cagr_n  = ((last_price / s_price) ** (1 / lookback) - 1) * 100
                    total_n = (last_price / s_price - 1) * 100
                    lines.append(
                        f"\n{lookback}-Year CAGR ({target}–{last_yr}):"
                        f"  {cagr_n:.2f}%/yr  (total {total_n:+.1f}%,"
                        f"  ${s_price:,.0f} → ${last_price:,.0f})"
                    )

            return Document(page_content="\n".join(lines), metadata={"type": "appreciation"})
        except Exception:
            return None

    def _seasonal_price_doc(self) -> Document | None:
        """
        Average sale price and sales volume broken down by month and season.
        """
        try:
            df = self.loader.df
            if "Sales_Volume" not in df.columns or "Sales" not in df.columns:
                return None

            work = df[df["Sales"] > 0].copy()
            work["avg_price"] = work["Sales_Volume"] / work["Sales"]

            monthly = work.groupby(work.index.month).agg(
                avg_price=("avg_price", "mean"),
                avg_sales=("Sales",     "mean"),
            )

            _MNAMES = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                       7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
            _SEASONS = [
                ("Spring (Mar–May)", [3, 4, 5]),
                ("Summer (Jun–Aug)", [6, 7, 8]),
                ("Fall   (Sep–Nov)", [9, 10, 11]),
                ("Winter (Dec–Feb)", [12, 1, 2]),
            ]

            lines = [
                "Dayton MSA – Average Sale Price & Activity by Season and Month",
                "(Historical averages across all years in dataset)",
                "-" * 65,
                "Seasonal Summary:",
            ]
            for s_name, months in _SEASONS:
                avail = [m for m in months if m in monthly.index]
                if avail:
                    s_price = monthly.loc[avail, "avg_price"].mean()
                    s_sales = monthly.loc[avail, "avg_sales"].mean()
                    lines.append(f"  {s_name}:  ${s_price:,.0f} avg price  |  {s_sales:.0f} avg sales/mo")

            best_p  = int(monthly["avg_price"].idxmax())
            worst_p = int(monthly["avg_price"].idxmin())
            best_v  = int(monthly["avg_sales"].idxmax())
            worst_v = int(monthly["avg_sales"].idxmin())

            lines += [
                "",
                "Monthly Detail:",
            ]
            for m, row in monthly.iterrows():
                lines.append(
                    f"  {_MNAMES[m]:<11}: ${row['avg_price']:,.0f} avg price"
                    f"  |  {row['avg_sales']:.0f} avg sales"
                )

            lines += [
                "",
                f"Highest-price month:  {_MNAMES[best_p]} (${monthly.loc[best_p,'avg_price']:,.0f})",
                f"Lowest-price month:   {_MNAMES[worst_p]} (${monthly.loc[worst_p,'avg_price']:,.0f})",
                f"Busiest sales month:  {_MNAMES[best_v]} ({monthly.loc[best_v,'avg_sales']:.0f} avg sales)",
                f"Slowest sales month:  {_MNAMES[worst_v]} ({monthly.loc[worst_v,'avg_sales']:.0f} avg sales)",
            ]

            return Document(page_content="\n".join(lines), metadata={"type": "seasonal_price"})
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
