"""
RefiAnalyzer
Identifies historical refinancing opportunity windows and calculates monthly
payment savings for configurable loan amounts.

Fixes vs the original notebooks
────────────────────────────────
• Refi windows are computed against the CURRENT rate (dynamic), not a fixed
  hardcoded rate from an earlier run.
• Loan scenarios are driven by config constants, not buried magic numbers.
• Monthly savings use the standard P&I amortisation formula.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from src import config


class RefiAnalyzer:
    """
    Parameters
    ----------
    df : Merged DataFrame from RealEstateDataLoader; must contain '30yrFRM'.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if "30yrFRM" not in df.columns:
            raise ValueError("DataFrame must contain a '30yrFRM' column.")
        self.rates: pd.Series = df["30yrFRM"].dropna()
        self.current_rate: float = float(self.rates.iloc[-1])

    # ── Core calculations ─────────────────────────────────────────────────────

    @staticmethod
    def monthly_payment(principal: float, annual_rate_pct: float, term_years: int = 30) -> float:
        """Standard fixed-rate P&I monthly payment."""
        r = annual_rate_pct / 100 / 12
        n = term_years * 12
        if r == 0:
            return principal / n
        return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)

    def monthly_savings(
        self,
        principal: float,
        old_rate_pct: float,
        new_rate_pct: float,
        term_years: int = 30,
    ) -> float:
        """Monthly P&I savings when refinancing from old_rate to new_rate."""
        old = self.monthly_payment(principal, old_rate_pct, term_years)
        new = self.monthly_payment(principal, new_rate_pct, term_years)
        return round(old - new, 2)

    # ── Window detection ──────────────────────────────────────────────────────

    def find_refi_windows(
        self,
        threshold_pp: float = config.REFI_THRESHOLD_PP,
        reference_rate: float | None = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of months where historical rates exceeded
        (reference_rate + threshold_pp).

        A borrower who locked in at reference_rate (default = current rate)
        would benefit from refinancing in any returned month where the rate
        was higher — i.e., those borrowers now have an incentive to refi down
        to today's rates.

        Columns: date, rate, excess_pp, savings_150k, savings_300k
        """
        ref = reference_rate if reference_rate is not None else self.current_rate
        trigger = ref + threshold_pp

        windows = self.rates[self.rates > trigger].copy().reset_index()
        windows.columns = ["date", "rate"]
        windows["excess_pp"] = (windows["rate"] - ref).round(2)

        for loan in config.REFI_LOAN_AMOUNTS:
            key = f"savings_{int(loan/1000)}k"
            windows[key] = windows["rate"].apply(
                lambda r: self.monthly_savings(loan, r, ref)
            )

        return windows.reset_index(drop=True)

    # ── Summary ───────────────────────────────────────────────────────────────

    def generate_refi_summary(
        self,
        threshold_pp: float = config.REFI_THRESHOLD_PP,
    ) -> str:
        """Narrative summary suitable for the AI agent knowledge base."""
        windows = self.find_refi_windows(threshold_pp)
        n_months = len(windows)

        if n_months == 0:
            return (
                f"Refinancing Opportunity Analysis (threshold: {threshold_pp} pp above current rate "
                f"{self.current_rate}%):\n"
                f"No months in the historical record show rates significantly above today's rate. "
                "Current borrowers are likely already near historical lows for their era."
            )

        most_recent = windows.sort_values("date").iloc[-1]
        loan_lines = "\n".join(
            f"  - ${loan:,} loan: save ~${self.monthly_savings(loan, most_recent['rate'], self.current_rate):,.0f}/month"
            for loan in config.REFI_LOAN_AMOUNTS
        )

        return (
            f"Refinancing Opportunity Analysis (current rate: {self.current_rate}%):\n"
            f"- Threshold: {threshold_pp} percentage points above current rate "
            f"({self.current_rate + threshold_pp:.2f}%)\n"
            f"- Months in history where rates exceeded this threshold: {n_months}\n"
            f"- Most recent such month: {most_recent['date'].strftime('%B %Y')} "
            f"(rate: {most_recent['rate']:.2f}%)\n"
            f"- Estimated monthly savings for borrowers refinancing from that rate:\n"
            f"{loan_lines}\n"
            f"- Annual savings for a $300k borrower: "
            f"${self.monthly_savings(300_000, most_recent['rate'], self.current_rate) * 12:,.0f}\n"
            f"Note: Savings are before closing costs. Break-even typically 18–36 months."
        )

    def get_current_refi_opportunities(self) -> List[Tuple[float, float, float]]:
        """
        Returns list of (loan_amount, old_rate, monthly_savings) for each loan size
        using the most recent above-threshold rate as the 'old' rate.
        """
        windows = self.find_refi_windows()
        if windows.empty:
            return []

        most_recent_rate = float(windows.sort_values("date").iloc[-1]["rate"])
        return [
            (loan, most_recent_rate, self.monthly_savings(loan, most_recent_rate, self.current_rate))
            for loan in config.REFI_LOAN_AMOUNTS
        ]
