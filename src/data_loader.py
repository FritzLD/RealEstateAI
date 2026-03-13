"""
RealEstateDataLoader
Loads, cleans, and merges the three data sources (market stats, mortgage rates,
economic indicators).  Also generates LangChain Documents for the RAG knowledge base.

Fixes vs the original notebooks
────────────────────────────────
• DatetimeIndex frequency explicitly set to 'MS' (eliminates VAR/SARIMAX warnings)
• String→float conversion is robust (handles commas, dollar signs, empty strings)
• Exogenous variable selection uses iterative VIF instead of manual column drops
• Moving-average windows read from config constants (MA_SHORT, MA_LONG)
• Derived metrics (Disparity, Disparity%) recalculated cleanly in one place
"""

from __future__ import annotations

import re
import warnings
from typing import List

import numpy as np
import pandas as pd
from langchain_core.documents import Document
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src import config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_float(series: pd.Series) -> pd.Series:
    """Strip commas / dollar signs / % signs and coerce to float."""
    return (
        series.astype(str)
        .str.replace(r"[$,%]", "", regex=True)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("", np.nan)
        .astype(float)
    )


def select_exog_by_vif(df: pd.DataFrame, threshold: float = config.VIF_THRESHOLD) -> pd.DataFrame:
    """
    Iteratively drop columns whose VIF exceeds *threshold*.
    Returns a DataFrame containing only the surviving columns.
    VIF requires a constant column; one is added temporarily.
    """
    cols = list(df.columns)
    while True:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = df[cols].assign(_const=1.0).dropna()
            vifs = {
                c: variance_inflation_factor(X.values, i)
                for i, c in enumerate(X.columns[:-1])  # exclude _const
            }
        worst_col = max(vifs, key=vifs.get)  # type: ignore[arg-type]
        if vifs[worst_col] > threshold:
            cols.remove(worst_col)
        else:
            break
    return df[cols]


# ── Main class ────────────────────────────────────────────────────────────────

class RealEstateDataLoader:
    """
    Loads, merges, and exposes the cleaned Dayton MSA real estate dataset.

    Usage
    -----
    loader = RealEstateDataLoader()
    df = loader.df          # fully merged DataFrame, monthly frequency
    docs = loader.generate_knowledge_documents()
    """

    def __init__(self) -> None:
        self.df: pd.DataFrame = self._load_and_merge()
        self.exog_cols: List[str] = self._identify_exog_cols()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_market_stats(self) -> pd.DataFrame:
        """Load statisticsday.csv – primary sales/listings data."""
        df = pd.read_csv(config.STATS_CSV)

        # Flexible column renaming: tolerate slight variations in source headers
        rename = {}
        for col in df.columns:
            lc = col.strip().lower()
            if "active" in lc:
                rename[col] = "Active"
            elif "sales" in lc and "volume" not in lc and "number" in lc:
                rename[col] = "Sales"
            elif "new" in lc and "listing" in lc:
                rename[col] = "New_Listings"
            elif "volume" in lc:
                rename[col] = "Sales_Volume"
            elif "expir" in lc:
                rename[col] = "Expired_Listings"
            elif "month" in lc or "date" in lc or "period" in lc:
                rename[col] = "Month"
        df = df.rename(columns=rename)

        # Parse date and set index with explicit monthly-start frequency
        df["Month"] = pd.to_datetime(df["Month"])
        df = df.set_index("Month").sort_index()
        df.index = df.index.to_period("M").to_timestamp("s")
        df.index.freq = "MS"

        # Convert all numeric columns
        for col in ["Active", "Sales", "New_Listings", "Sales_Volume", "Expired_Listings"]:
            if col in df.columns:
                df[col] = _to_float(df[col])

        return df[["Active", "Sales", "New_Listings", "Sales_Volume", "Expired_Listings"]].dropna(
            subset=["Active", "Sales"]
        )

    def _load_mortgage_rates(self) -> pd.DataFrame:
        """Load historicalweeklydata.xlsx – weekly Freddie Mac PMMS → monthly."""
        xl = pd.read_excel(config.RATES_XLSX)

        # Find date column
        date_col = next(
            (c for c in xl.columns if re.search(r"date|week|period", c, re.I)), xl.columns[0]
        )
        xl[date_col] = pd.to_datetime(xl[date_col], errors="coerce")
        xl = xl.dropna(subset=[date_col]).set_index(date_col).sort_index()

        # Find 30-year rate column
        rate_col = next(
            (c for c in xl.columns if re.search(r"30", c)), xl.columns[0]
        )
        xl[rate_col] = _to_float(xl[rate_col])

        # Resample weekly → monthly average, align to month-start
        monthly = xl[[rate_col]].resample("MS").mean()
        monthly.columns = ["30yrFRM"]
        monthly.index.freq = "MS"
        return monthly

    def _load_economic_indicators(self) -> pd.DataFrame:
        """Load inflation.xlsx – CPI, Inflation, Unemployment, Sentiment, Geopolitical."""
        xl = pd.read_excel(config.INFLATION_XLSX)

        # Find date column
        date_col = next(
            (c for c in xl.columns if re.search(r"date|month|period|year", c, re.I)), xl.columns[0]
        )
        xl[date_col] = pd.to_datetime(xl[date_col], errors="coerce")
        xl = xl.dropna(subset=[date_col]).set_index(date_col).sort_index()
        xl.index = xl.index.to_period("M").to_timestamp("s")
        xl.index.freq = "MS"

        # Rename to standard names where identifiable
        rename = {}
        for col in xl.columns:
            lc = col.strip().lower()
            if "cpi" in lc:
                rename[col] = "CPI"
            elif "inflation" in lc:
                rename[col] = "Inflation_Rate"
            elif "unemploy" in lc:
                rename[col] = "Unemployment"
            elif "sentiment" in lc or "confidence" in lc:
                rename[col] = "Consumer_Sentiment_Index"
            elif "gpr" in lc or "geopolit" in lc:
                rename[col] = "GPRHC_USA"
        xl = xl.rename(columns=rename)

        # Convert to float
        for col in xl.columns:
            xl[col] = _to_float(xl[col])

        return xl

    def _load_and_merge(self) -> pd.DataFrame:
        """Merge all three sources on the monthly DatetimeIndex."""
        market = self._load_market_stats()
        rates  = self._load_mortgage_rates()
        econ   = self._load_economic_indicators()

        df = market.join(rates, how="left").join(econ, how="left")
        df.index.freq = "MS"

        # Forward-fill gaps from imperfect alignment (up to 2 months)
        df = df.ffill(limit=2)

        # ── Derived columns ───────────────────────────────────────────────────
        df["Disparity"]    = df["Active"] - df["Sales"]
        df["Disparity_pct"] = ((df["Active"] - df["Sales"]) / df["Active"].replace(0, np.nan)) * 100
        df[f"MA_{config.MA_SHORT}"] = df["Sales"].rolling(config.MA_SHORT).mean()
        df[f"MA_{config.MA_LONG}"]  = df["Sales"].rolling(config.MA_LONG).mean()
        df["Disparity_MA"] = df["Disparity_pct"].rolling(config.MA_SHORT).mean()

        return df

    # ── Exogenous variable selection ──────────────────────────────────────────

    def _identify_exog_cols(self) -> List[str]:
        """Return economic indicator columns that pass VIF screening."""
        candidates = [
            c for c in ["30yrFRM", "CPI", "Inflation_Rate", "Unemployment",
                         "Consumer_Sentiment_Index", "GPRHC_USA"]
            if c in self.df.columns
        ]
        if not candidates:
            return []
        subset = self.df[candidates].dropna()
        if subset.empty:
            return candidates
        survived = select_exog_by_vif(subset)
        return list(survived.columns)

    # ── Summary statistics ────────────────────────────────────────────────────

    def get_market_summary(self) -> dict:
        df = self.df
        latest = df.iloc[-1]
        prev_year = df.iloc[-13] if len(df) >= 13 else df.iloc[0]

        return {
            "data_through":      df.index[-1].strftime("%B %Y"),
            "data_from":         df.index[0].strftime("%B %Y"),
            "current_active":    int(latest.get("Active", 0)),
            "current_sales":     int(latest.get("Sales", 0)),
            "disparity_pct":     round(float(latest.get("Disparity_pct", 0)), 1),
            "current_rate":      round(float(latest.get("30yrFRM", 0)), 2),
            "yoy_sales_chg":     round(
                (latest.get("Sales", 0) - prev_year.get("Sales", 0))
                / max(prev_year.get("Sales", 1), 1) * 100, 1
            ),
            "yoy_active_chg":    round(
                (latest.get("Active", 0) - prev_year.get("Active", 0))
                / max(prev_year.get("Active", 1), 1) * 100, 1
            ),
            "avg_sales_12mo":    round(df["Sales"].iloc[-12:].mean(), 1),
            "avg_active_12mo":   round(df["Active"].iloc[-12:].mean(), 1),
            # Historical mean disparity used as balanced-market baseline
            "balanced_threshold_pct": round(float(df["Disparity_pct"].mean()), 1),
            "market_condition":  (
                "Seller's Market" if df["Disparity_pct"].iloc[-1] < df["Disparity_pct"].mean()
                else "Buyer's Market"
            ),
        }

    def get_disparity_stats(self) -> dict:
        df = self.df
        return {
            "current_disparity":      int(df["Disparity"].iloc[-1]),
            "current_disparity_pct":  round(float(df["Disparity_pct"].iloc[-1]), 1),
            "avg_disparity_pct_12mo": round(df["Disparity_pct"].iloc[-12:].mean(), 1),
            "max_disparity_pct":      round(float(df["Disparity_pct"].max()), 1),
            "min_disparity_pct":      round(float(df["Disparity_pct"].min()), 1),
            # Historical mean disparity % used as balanced-market baseline.
            # Computed from actual Dayton MSA data to remove long-run bias
            # rather than using an arbitrary fixed threshold.
            "balanced_threshold_pct": round(float(df["Disparity_pct"].mean()), 1),
            "market_condition":       (
                "Seller's Market" if df["Disparity_pct"].iloc[-1] < df["Disparity_pct"].mean()
                else "Buyer's Market"
            ),
        }

    def get_trend_stats(self) -> dict:
        df = self.df
        sales_6mo_avg  = df["Sales"].iloc[-6:].mean()
        sales_prior_6  = df["Sales"].iloc[-12:-6].mean()
        active_6mo_avg = df["Active"].iloc[-6:].mean()
        active_prior_6 = df["Active"].iloc[-12:-6].mean()

        return {
            "sales_trend_6mo":  round((sales_6mo_avg - sales_prior_6) / max(sales_prior_6, 1) * 100, 1),
            "active_trend_6mo": round((active_6mo_avg - active_prior_6) / max(active_prior_6, 1) * 100, 1),
            "peak_sales_month": df["Sales"].idxmax().strftime("%B %Y"),
            "peak_sales_value": int(df["Sales"].max()),
            "low_sales_month":  df["Sales"].idxmin().strftime("%B %Y"),
            "low_sales_value":  int(df["Sales"].min()),
        }

    def get_rate_stats(self) -> dict:
        if "30yrFRM" not in self.df.columns:
            return {}
        r = self.df["30yrFRM"].dropna()
        return {
            "current_rate":  round(float(r.iloc[-1]), 2),
            "rate_1yr_ago":  round(float(r.iloc[-13]) if len(r) >= 13 else float(r.iloc[0]), 2),
            "rate_peak":     round(float(r.max()), 2),
            "rate_peak_date": r.idxmax().strftime("%B %Y"),
            "rate_low":      round(float(r.min()), 2),
            "rate_low_date": r.idxmin().strftime("%B %Y"),
            "rate_5yr_avg":  round(float(r.iloc[-60:].mean()), 2) if len(r) >= 60 else round(float(r.mean()), 2),
        }

    # ── RAG document generation ───────────────────────────────────────────────

    def generate_knowledge_documents(self) -> List[Document]:
        """Convert all stats into LangChain Documents for vector-store ingestion."""
        docs: List[Document] = []

        # 1. Market overview
        s = self.get_market_summary()
        docs.append(Document(
            page_content=(
                f"Dayton MSA Real Estate Market Summary (data through {s['data_through']}):\n"
                f"- Current active listings: {s['current_active']:,}\n"
                f"- Current monthly sales: {s['current_sales']:,}\n"
                f"- Market disparity (Active-Sales / Active): {s['disparity_pct']}%\n"
                f"- Current 30-year fixed mortgage rate: {s['current_rate']}%\n"
                f"- Year-over-year sales change: {s['yoy_sales_chg']:+}%\n"
                f"- Year-over-year active listings change: {s['yoy_active_chg']:+}%\n"
                f"- 12-month average monthly sales: {s['avg_sales_12mo']:,}\n"
                f"- 12-month average active listings: {s['avg_active_12mo']:,}\n"
                f"- Historical data range: {s['data_from']} to {s['data_through']}"
            ),
            metadata={"type": "market_overview"},
        ))

        # 2. Market disparity / inventory
        d = self.get_disparity_stats()
        docs.append(Document(
            page_content=(
                f"Dayton MSA Inventory-Sales Disparity Analysis:\n"
                f"- Current disparity: {d['current_disparity']:,} units (Active minus Sales)\n"
                f"- Current disparity %: {d['current_disparity_pct']}%\n"
                f"- 12-month average disparity %: {d['avg_disparity_pct_12mo']}%\n"
                f"- Market condition: {d['market_condition']}\n"
                f"- Balanced market baseline: {d['balanced_threshold_pct']}% (historical mean disparity)\n"
                f"  (Below {d['balanced_threshold_pct']}% = tighter than historical average = Seller's Market)\n"
                f"- All-time high disparity %: {d['max_disparity_pct']}%\n"
                f"- All-time low disparity %: {d['min_disparity_pct']}%"
            ),
            metadata={"type": "inventory_disparity"},
        ))

        # 3. Trends
        t = self.get_trend_stats()
        docs.append(Document(
            page_content=(
                f"Dayton MSA Market Trend Analysis:\n"
                f"- Sales trend (last 6 months vs prior 6): {t['sales_trend_6mo']:+}%\n"
                f"- Active listings trend (last 6 months vs prior 6): {t['active_trend_6mo']:+}%\n"
                f"- Peak sales month: {t['peak_sales_month']} ({t['peak_sales_value']:,} units)\n"
                f"- Slowest sales month: {t['low_sales_month']} ({t['low_sales_value']:,} units)"
            ),
            metadata={"type": "market_trends"},
        ))

        # 4. Mortgage rates
        r = self.get_rate_stats()
        if r:
            docs.append(Document(
                page_content=(
                    f"30-Year Fixed Mortgage Rate History (Dayton MSA / National):\n"
                    f"- Current rate: {r['current_rate']}%\n"
                    f"- Rate one year ago: {r['rate_1yr_ago']}%\n"
                    f"- All-time high in dataset: {r['rate_peak']}% ({r['rate_peak_date']})\n"
                    f"- All-time low in dataset: {r['rate_low']}% ({r['rate_low_date']})\n"
                    f"- 5-year average rate: {r['rate_5yr_avg']}%"
                ),
                metadata={"type": "mortgage_rates"},
            ))

        # 5. Seasonal patterns
        df = self.df
        monthly_avg = df.groupby(df.index.month)["Sales"].mean()
        best_month  = monthly_avg.idxmax()
        worst_month = monthly_avg.idxmin()
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        docs.append(Document(
            page_content=(
                f"Dayton MSA Seasonal Real Estate Patterns:\n"
                f"- Historically strongest sales month: {month_names[best_month]} "
                f"(avg {monthly_avg[best_month]:.0f} sales)\n"
                f"- Historically slowest sales month: {month_names[worst_month]} "
                f"(avg {monthly_avg[worst_month]:.0f} sales)\n"
                f"- Spring (Mar–May) average sales: {monthly_avg[[3,4,5]].mean():.0f}/month\n"
                f"- Summer (Jun–Aug) average sales: {monthly_avg[[6,7,8]].mean():.0f}/month\n"
                f"- Fall (Sep–Nov) average sales:   {monthly_avg[[9,10,11]].mean():.0f}/month\n"
                f"- Winter (Dec–Feb) average sales:  {monthly_avg[[12,1,2]].mean():.0f}/month"
            ),
            metadata={"type": "seasonal_patterns"},
        ))

        # 6. Complete year-by-year historical record — ALL years in the dataset
        all_years = sorted(df.index.year.unique())
        rate_col_exists = "30yrFRM" in df.columns
        yoy_rows = []
        for yr in all_years:
            yd = df[df.index.year == yr]
            if len(yd) < 6:
                continue
            rate_str = (
                f", avg rate {yd['30yrFRM'].mean():.2f}%"
                if rate_col_exists and yd["30yrFRM"].notna().any()
                else ""
            )
            disparity_str = (
                f", avg disparity {yd['Disparity_pct'].mean():.1f}%"
                if "Disparity_pct" in yd.columns and yd["Disparity_pct"].notna().any()
                else ""
            )
            price_str = ""
            if "Sales_Volume" in yd.columns:
                yr_sales = yd["Sales"].sum()
                if yr_sales > 0:
                    avg_price = yd["Sales_Volume"].sum() / yr_sales
                    price_str = f", avg sale price ${avg_price:,.0f}"
            yoy_rows.append(
                f"  {yr}: avg monthly sales {yd['Sales'].mean():.0f}, "
                f"avg active listings {yd['Active'].mean():.0f}"
                f"{rate_str}{disparity_str}{price_str}"
            )
        docs.append(Document(
            page_content=(
                f"Dayton MSA Complete Annual Market History ({all_years[0]}–{all_years[-1]}):\n"
                + "\n".join(yoy_rows)
            ),
            metadata={"type": "full_annual_history"},
        ))

        # 7. Era / period summaries for contextual questions like "10 years ago"
        era_lines = []
        era_definitions = [
            ("Post-financial-crisis recovery", 2010, 2012),
            ("Market stabilization",           2013, 2015),
            ("Pre-pandemic growth",            2016, 2019),
            ("COVID-19 disruption",            2020, 2021),
            ("Post-COVID rate-hike era",       2022, 2023),
            ("Recent market",                  2024, all_years[-1]),
        ]
        for label, yr_start, yr_end in era_definitions:
            era_df = df[(df.index.year >= yr_start) & (df.index.year <= yr_end)]
            if era_df.empty:
                continue
            rate_str = (
                f", avg 30yr rate {era_df['30yrFRM'].mean():.2f}%"
                if rate_col_exists and era_df["30yrFRM"].notna().any()
                else ""
            )
            era_lines.append(
                f"  {label} ({yr_start}–{yr_end}): "
                f"avg sales {era_df['Sales'].mean():.0f}/mo, "
                f"avg active {era_df['Active'].mean():.0f}/mo"
                f"{rate_str}"
            )
        docs.append(Document(
            page_content=(
                "Dayton MSA Market Eras – Historical Period Summaries:\n"
                + "\n".join(era_lines)
            ),
            metadata={"type": "market_eras"},
        ))

        # 8. Mortgage rate full annual history
        if rate_col_exists:
            rate_rows = []
            for yr in all_years:
                yd = df[df.index.year == yr]
                if yd["30yrFRM"].notna().any():
                    rate_rows.append(
                        f"  {yr}: avg {yd['30yrFRM'].mean():.2f}%, "
                        f"range {yd['30yrFRM'].min():.2f}%–{yd['30yrFRM'].max():.2f}%"
                    )
            docs.append(Document(
                page_content=(
                    f"30-Year Fixed Mortgage Rate Annual History ({all_years[0]}–{all_years[-1]}):\n"
                    + "\n".join(rate_rows)
                ),
                metadata={"type": "rate_annual_history"},
            ))

        # 9. Economic context — current snapshot
        econ_cols = [c for c in ["CPI", "Inflation_Rate", "Unemployment",
                                  "Consumer_Sentiment_Index"] if c in df.columns]
        if econ_cols:
            latest = df[econ_cols].iloc[-1].dropna()
            lines = [f"  - {col}: {val:.2f}" for col, val in latest.items()]
            docs.append(Document(
                page_content=(
                    f"Current Economic Indicators ({df.index[-1].strftime('%B %Y')}):\n"
                    + "\n".join(lines)
                ),
                metadata={"type": "economic_indicators"},
            ))

        return docs
