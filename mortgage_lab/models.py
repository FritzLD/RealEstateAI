"""Structured inputs and results for the mortgage engine.

Inputs are captured in :class:`MortgageScenario`, which validates itself on
construction. Outputs are captured in :class:`MortgageResult`, which can be
converted to a ``dict``, JSON, or pandas DataFrames for display or tool use.

Units used throughout the engine:

* Currency values are US dollars as ``float``.
* Rates (interest rate and program fee rates) are **percentages**, so
  6.375 means 6.375% and 1.75 means 1.75%.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from typing import Any

import pandas as pd

# Largest term the engine accepts. The UI currently offers 15 and 30 years;
# other positive terms up to this limit work in the engine without changes.
MAX_TERM_YEARS = 50

# Upper sanity bound for user-entered percentages. This is an input guard,
# not a lending rule.
MAX_INTEREST_RATE_PCT = 30.0
MAX_PROGRAM_FEE_RATE_PCT = 10.0


class ScenarioValidationError(ValueError):
    """Raised when a scenario contains values the engine cannot accept.

    The message is written for end users and can be shown directly in the UI.
    """


class LoanProgram(str, Enum):
    """Supported loan programs."""

    CONVENTIONAL = "Conventional"
    FHA = "FHA"
    VA = "VA"
    USDA = "USDA"

    @classmethod
    def from_value(cls, value: "LoanProgram | str") -> "LoanProgram":
        """Accept an enum member or its display name (case-insensitive)."""
        if isinstance(value, cls):
            return value
        for member in cls:
            if str(value).strip().lower() in (member.value.lower(), member.name.lower()):
                return member
        valid = ", ".join(m.value for m in cls)
        raise ScenarioValidationError(f"Unknown loan program '{value}'. Choose one of: {valid}.")


@dataclass(frozen=True)
class MortgageScenario:
    """A single set of financing assumptions.

    Program-specific rate fields are optional overrides. When left as ``None``
    the program schedule in :mod:`mortgage_lab.loan_programs` is used (including any
    down-payment tiers). Fields that do
    not apply to the selected program are ignored.

    Attributes:
        loan_program: Conventional, FHA, VA, or USDA (enum or display name).
        purchase_price: Purchase price / property value in dollars.
        down_payment: Down payment in dollars.
        interest_rate: Annual note rate as a percentage (user modeling assumption).
        term_years: Loan term in years (UI offers 15 and 30).
        first_payment_date: Optional first payment date. Defaults to the first
            day of the following month when omitted.
        finance_program_fee: Whether the upfront program fee (FHA UFMIP, VA
            funding fee, USDA upfront guarantee fee) is added to the loan.
        va_funding_fee_exempt: VA only. If True the funding fee is $0.
        va_funding_fee_rate: VA funding fee rate override (percent).
        fha_upfront_mip_rate: FHA upfront MIP rate override (percent).
        fha_annual_mip_rate: FHA annual MIP rate override (percent).
        usda_upfront_fee_rate: USDA upfront guarantee fee rate override (percent).
        usda_annual_fee_rate: USDA annual guarantee fee rate override (percent).
        extra_monthly_principal: Optional additional principal paid each month.
    """

    loan_program: LoanProgram | str
    purchase_price: float
    down_payment: float
    interest_rate: float
    term_years: int
    first_payment_date: date | None = None

    finance_program_fee: bool = True

    va_funding_fee_exempt: bool = False
    va_funding_fee_rate: float | None = None

    fha_upfront_mip_rate: float | None = None
    fha_annual_mip_rate: float | None = None

    usda_upfront_fee_rate: float | None = None
    usda_annual_fee_rate: float | None = None

    extra_monthly_principal: float = 0.0

    def __post_init__(self) -> None:
        # Frozen dataclass: normalise the program with object.__setattr__.
        object.__setattr__(self, "loan_program", LoanProgram.from_value(self.loan_program))
        self.validate()

    @property
    def number_of_payments(self) -> int:
        """Total scheduled monthly payments (term in years x 12)."""
        return int(self.term_years) * 12

    def validate(self) -> None:
        """Raise :class:`ScenarioValidationError` if any input is invalid."""
        if self.purchase_price is None or self.purchase_price <= 0:
            raise ScenarioValidationError("Purchase price must be greater than $0.")
        if self.down_payment is None or self.down_payment < 0:
            raise ScenarioValidationError("Down payment cannot be negative.")
        if self.down_payment > self.purchase_price:
            raise ScenarioValidationError("Down payment cannot exceed the purchase price.")
        if self.interest_rate is None or self.interest_rate < 0:
            raise ScenarioValidationError("Interest rate cannot be negative.")
        if self.interest_rate > MAX_INTEREST_RATE_PCT:
            raise ScenarioValidationError(
                f"Interest rate above {MAX_INTEREST_RATE_PCT:g}% is outside the supported range."
            )
        if not isinstance(self.term_years, int) or isinstance(self.term_years, bool):
            raise ScenarioValidationError("Loan term must be a whole number of years.")
        if not 0 < self.term_years <= MAX_TERM_YEARS:
            raise ScenarioValidationError(
                f"Loan term must be between 1 and {MAX_TERM_YEARS} years."
            )
        if self.extra_monthly_principal is None or self.extra_monthly_principal < 0:
            raise ScenarioValidationError("Extra monthly principal cannot be negative.")

        rate_fields = {
            "VA funding fee rate": self.va_funding_fee_rate,
            "FHA upfront MIP rate": self.fha_upfront_mip_rate,
            "FHA annual MIP rate": self.fha_annual_mip_rate,
            "USDA upfront guarantee fee rate": self.usda_upfront_fee_rate,
            "USDA annual guarantee fee rate": self.usda_annual_fee_rate,
        }
        for label, value in rate_fields.items():
            if value is None:
                continue
            if value < 0:
                raise ScenarioValidationError(f"{label} cannot be negative.")
            if value > MAX_PROGRAM_FEE_RATE_PCT:
                raise ScenarioValidationError(
                    f"{label} above {MAX_PROGRAM_FEE_RATE_PCT:g}% is outside the supported range."
                )

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python representation (JSON-safe)."""
        data = asdict(self)
        data["loan_program"] = self.loan_program.value
        data["first_payment_date"] = (
            self.first_payment_date.isoformat() if self.first_payment_date else None
        )
        return data


@dataclass(frozen=True)
class AmortizationRow:
    """One monthly payment in the amortization schedule.

    ``principal`` includes any extra principal paid that month (also shown on
    its own in ``extra_principal``). ``program_fee`` is the recurring FHA MIP or
    USDA annual fee; it is never part of principal, interest, or balance.
    """

    payment_number: int
    payment_date: date
    beginning_balance: float
    scheduled_pi: float
    principal: float
    interest: float
    extra_principal: float
    program_fee: float
    total_payment: float
    ending_balance: float
    cumulative_principal: float
    cumulative_interest: float


SCHEDULE_COLUMN_LABELS: dict[str, str] = {
    "payment_number": "Payment #",
    "payment_date": "Payment Date",
    "beginning_balance": "Beginning Balance",
    "scheduled_pi": "P&I Payment",
    "principal": "Principal",
    "interest": "Interest",
    "extra_principal": "Extra Principal",
    "program_fee": "Program Fee",
    "total_payment": "Total Loan Payment",
    "ending_balance": "Ending Balance",
    "cumulative_principal": "Cumulative Principal",
    "cumulative_interest": "Cumulative Interest",
}


@dataclass(frozen=True)
class MortgageResult:
    """Deterministic output of :func:`mortgage_lab.mortgage_calculator.calculate_mortgage`.

    All currency values are rounded to cents. ``monthly_pi`` is the scheduled
    payment; the final scheduled payment can differ by a few cents because it
    is adjusted to close the balance at exactly $0.00.
    """

    loan_program: str
    purchase_price: float
    down_payment: float
    down_payment_pct: float
    base_loan_amount: float

    upfront_fee_rate: float
    upfront_program_fee: float
    financed_program_fee: float
    upfront_fee_paid_separately: float

    total_financed_loan_amount: float

    interest_rate: float
    term_years: int
    number_of_payments: int

    monthly_pi: float
    annual_program_fee_rate: float
    monthly_program_fee: float
    estimated_monthly_loan_payment: float

    total_principal: float
    total_interest: float

    first_payment_date: date
    payoff_date: date | None

    extra_monthly_principal: float = 0.0
    upfront_rate_source: str = ""
    annual_rate_source: str = ""
    program_assumptions_verified: bool = True
    notices: tuple[str, ...] = ()
    amortization_schedule: tuple[AmortizationRow, ...] = field(default=(), repr=False)

    @property
    def payments_made(self) -> int:
        """Number of payments actually in the schedule (fewer with extra principal)."""
        return len(self.amortization_schedule)

    def to_dict(self, include_schedule: bool = False) -> dict[str, Any]:
        """Plain-Python representation with ISO dates (JSON-safe)."""
        data = asdict(self)
        data.pop("amortization_schedule")
        data["first_payment_date"] = self.first_payment_date.isoformat()
        data["payoff_date"] = self.payoff_date.isoformat() if self.payoff_date else None
        data["notices"] = list(self.notices)
        data["payments_made"] = self.payments_made
        if include_schedule:
            data["amortization_schedule"] = [
                {**asdict(row), "payment_date": row.payment_date.isoformat()}
                for row in self.amortization_schedule
            ]
        return data

    def to_json(self, include_schedule: bool = False, indent: int | None = 2) -> str:
        """JSON string of :meth:`to_dict`."""
        return json.dumps(self.to_dict(include_schedule=include_schedule), indent=indent)

    def schedule_dataframe(self, display_labels: bool = False) -> pd.DataFrame:
        """Amortization schedule as a DataFrame (one row per payment)."""
        columns = list(SCHEDULE_COLUMN_LABELS)
        df = pd.DataFrame([asdict(row) for row in self.amortization_schedule], columns=columns)
        if display_labels:
            df = df.rename(columns=SCHEDULE_COLUMN_LABELS)
        return df

    def summary_dataframe(self) -> pd.DataFrame:
        """Single-row DataFrame of the summary fields (no schedule)."""
        return pd.DataFrame([self.to_dict()])
