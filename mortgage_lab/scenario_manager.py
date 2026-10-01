"""Scenario comparison, deltas, and rate sensitivity (no Streamlit dependency)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import pandas as pd

from mortgage_lab.formatting import (
    format_currency,
    format_date,
    format_percentage,
    format_rate,
    format_term,
)
from mortgage_lab.models import MortgageResult, MortgageScenario
from mortgage_lab.mortgage_calculator import calculate_mortgage

DEFAULT_RATE_OFFSETS: tuple[float, ...] = (-1.00, -0.50, -0.25, 0.0, 0.25, 0.50, 1.00)

# (row label, result attribute, formatter)
COMPARISON_FIELDS: tuple[tuple[str, str, callable], ...] = (
    ("Loan Program", "loan_program", str),
    ("Purchase Price", "purchase_price", format_currency),
    ("Down Payment $", "down_payment", format_currency),
    ("Down Payment %", "down_payment_pct", format_percentage),
    ("Interest Rate", "interest_rate", format_rate),
    ("Term", "term_years", format_term),
    ("Base Loan Amount", "base_loan_amount", format_currency),
    ("Upfront Program Fee", "upfront_program_fee", format_currency),
    ("Financed Program Fee", "financed_program_fee", format_currency),
    ("Total Financed Loan", "total_financed_loan_amount", format_currency),
    ("Monthly P&I", "monthly_pi", format_currency),
    ("Monthly Program Charge", "monthly_program_fee", format_currency),
    ("Estimated Monthly Loan Payment", "estimated_monthly_loan_payment", format_currency),
    ("Total Interest", "total_interest", format_currency),
    ("Payoff Date", "payoff_date", format_date),
)


@dataclass(frozen=True)
class ScenarioDelta:
    """Factual differences: ``current`` minus ``baseline`` (positive = current is higher)."""

    monthly_payment_difference: float
    down_payment_difference: float
    financed_loan_difference: float
    total_interest_difference: float

    def to_dict(self) -> dict[str, float]:
        return {
            "monthly_payment_difference": self.monthly_payment_difference,
            "down_payment_difference": self.down_payment_difference,
            "financed_loan_difference": self.financed_loan_difference,
            "total_interest_difference": self.total_interest_difference,
        }


def compute_deltas(baseline: MortgageResult, current: MortgageResult) -> ScenarioDelta:
    """Differences between two results, rounded to cents."""

    def diff(attr: str) -> float:
        return round(getattr(current, attr) - getattr(baseline, attr), 2)

    return ScenarioDelta(
        monthly_payment_difference=diff("estimated_monthly_loan_payment"),
        down_payment_difference=diff("down_payment"),
        financed_loan_difference=diff("total_financed_loan_amount"),
        total_interest_difference=diff("total_interest"),
    )


def comparison_table(results: dict[str, MortgageResult]) -> pd.DataFrame:
    """Side-by-side formatted comparison. Columns are scenario labels, in order."""
    data = {
        label: [fmt(getattr(result, attr)) for _, attr, fmt in COMPARISON_FIELDS]
        for label, result in results.items()
    }
    return pd.DataFrame(data, index=[row for row, _, _ in COMPARISON_FIELDS])


def delta_table(deltas: dict[str, ScenarioDelta]) -> pd.DataFrame:
    """Raw-number delta table; columns are comparison labels (e.g. 'Current − A')."""
    rows = {
        "Monthly Payment Difference": "monthly_payment_difference",
        "Down Payment Difference": "down_payment_difference",
        "Financed Loan Difference": "financed_loan_difference",
        "Total Interest Difference": "total_interest_difference",
    }
    return pd.DataFrame(
        {label: [getattr(d, attr) for attr in rows.values()] for label, d in deltas.items()},
        index=list(rows),
    )


def rate_sensitivity(
    scenario: MortgageScenario,
    offsets: tuple[float, ...] = DEFAULT_RATE_OFFSETS,
    today: date | None = None,
) -> pd.DataFrame:
    """Monthly P&I across rate offsets, all other assumptions held constant.

    Offsets producing a negative rate are skipped. Each row is a full engine
    calculation, so the 0.00 offset row matches the main result exactly.
    """
    base = calculate_mortgage(scenario, today=today)
    rows = []
    for offset in offsets:
        rate = round(scenario.interest_rate + offset, 6)
        if rate < 0:
            continue
        result = calculate_mortgage(replace(scenario, interest_rate=rate), today=today)
        rows.append(
            {
                "rate_offset": offset,
                "interest_rate": rate,
                "monthly_pi": result.monthly_pi,
                "estimated_monthly_loan_payment": result.estimated_monthly_loan_payment,
                "pi_change_vs_current": round(result.monthly_pi - base.monthly_pi, 2),
                "total_interest": result.total_interest,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ExtraPaymentImpact:
    """Effect of extra monthly principal versus the scheduled loan."""

    extra_monthly_principal: float
    scheduled_payoff_date: date | None
    new_payoff_date: date | None
    months_saved: int
    interest_saved: float


def extra_payment_impact(scenario: MortgageScenario, today: date | None = None) -> ExtraPaymentImpact:
    """Compare the scenario with and without its extra monthly principal."""
    scheduled = calculate_mortgage(replace(scenario, extra_monthly_principal=0.0), today=today)
    accelerated = calculate_mortgage(scenario, today=today)
    return ExtraPaymentImpact(
        extra_monthly_principal=scenario.extra_monthly_principal,
        scheduled_payoff_date=scheduled.payoff_date,
        new_payoff_date=accelerated.payoff_date,
        months_saved=scheduled.payments_made - accelerated.payments_made,
        interest_saved=round(scheduled.total_interest - accelerated.total_interest, 2),
    )
