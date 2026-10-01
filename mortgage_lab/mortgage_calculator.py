"""Core mortgage calculations.

The orchestration in :func:`calculate_mortgage` is shared by every program:

    base loan  ->  program fees (loan_programs)  ->  financed principal
               ->  monthly P&I (formula)  ->  amortization schedule
               ->  MortgageResult

Program-specific rules live only in :mod:`mortgage_lab.loan_programs`.
"""

from __future__ import annotations

from datetime import date

from mortgage_lab.amortization import build_schedule, default_first_payment_date, to_cents
from mortgage_lab.loan_programs import calculate_program_fees
from mortgage_lab.models import MortgageResult, MortgageScenario, ScenarioValidationError

MONTHS_PER_YEAR = 12


def monthly_rate(annual_rate_pct: float) -> float:
    """Convert an annual percentage rate to a monthly decimal rate."""
    return annual_rate_pct / 100 / MONTHS_PER_YEAR


def monthly_payment(principal: float, annual_rate_pct: float, number_of_payments: int) -> float:
    """Fixed-rate monthly principal and interest at full precision.

    Uses ``M = P * r(1+r)^n / ((1+r)^n - 1)`` where ``r`` is the monthly rate.
    At 0% interest the payment is ``P / n``. A zero principal returns 0.
    """
    if number_of_payments <= 0:
        raise ScenarioValidationError("Number of payments must be greater than zero.")
    if annual_rate_pct < 0:
        raise ScenarioValidationError("Interest rate cannot be negative.")
    if principal <= 0:
        return 0.0
    r = monthly_rate(annual_rate_pct)
    if r == 0:
        return principal / number_of_payments
    growth = (1 + r) ** number_of_payments
    return principal * r * growth / (growth - 1)


def down_payment_to_pct(purchase_price: float, down_payment: float) -> float:
    """Down payment dollars -> percentage of purchase price."""
    if purchase_price <= 0:
        raise ScenarioValidationError("Purchase price must be greater than $0.")
    return down_payment / purchase_price * 100


def pct_to_down_payment(purchase_price: float, down_payment_pct: float) -> float:
    """Down payment percentage -> dollars, rounded to cents."""
    if purchase_price <= 0:
        raise ScenarioValidationError("Purchase price must be greater than $0.")
    if down_payment_pct < 0:
        raise ScenarioValidationError("Down payment cannot be negative.")
    if down_payment_pct > 100:
        raise ScenarioValidationError("Down payment cannot exceed the purchase price.")
    return float(to_cents(purchase_price * down_payment_pct / 100))


def base_loan_amount(purchase_price: float, down_payment: float) -> float:
    """Purchase price minus down payment, rounded to cents."""
    return float(to_cents(purchase_price - down_payment))


def calculate_mortgage(scenario: MortgageScenario, today: date | None = None) -> MortgageResult:
    """Calculate a complete, deterministic result for one scenario.

    Args:
        scenario: Validated financing assumptions.
        today: Reference date for the default first payment date. Supplying it
            makes results reproducible; it is ignored when the scenario has its
            own ``first_payment_date``.
    """
    base_loan = base_loan_amount(scenario.purchase_price, scenario.down_payment)
    fees = calculate_program_fees(scenario, base_loan)
    financed_principal = float(to_cents(base_loan + fees.financed_upfront_fee))

    n = scenario.number_of_payments
    pi = float(to_cents(monthly_payment(financed_principal, scenario.interest_rate, n)))
    first_date = scenario.first_payment_date or default_first_payment_date(today)

    schedule = build_schedule(
        principal=financed_principal,
        annual_rate_pct=scenario.interest_rate,
        number_of_payments=n,
        scheduled_pi=pi,
        first_payment_date=first_date,
        monthly_program_fee=fees.monthly_program_fee if financed_principal > 0 else 0.0,
        extra_monthly_principal=scenario.extra_monthly_principal,
    )

    total_interest = float(sum(to_cents(row.interest) for row in schedule))
    total_principal = float(sum(to_cents(row.principal) for row in schedule))
    monthly_fee = fees.monthly_program_fee if financed_principal > 0 else 0.0

    return MortgageResult(
        loan_program=scenario.loan_program.value,
        purchase_price=float(to_cents(scenario.purchase_price)),
        down_payment=float(to_cents(scenario.down_payment)),
        down_payment_pct=down_payment_to_pct(scenario.purchase_price, scenario.down_payment),
        base_loan_amount=base_loan,
        upfront_fee_rate=fees.upfront_fee_rate,
        upfront_program_fee=fees.upfront_fee,
        financed_program_fee=fees.financed_upfront_fee,
        upfront_fee_paid_separately=fees.upfront_fee_paid_separately,
        total_financed_loan_amount=financed_principal,
        interest_rate=scenario.interest_rate,
        term_years=scenario.term_years,
        number_of_payments=n,
        monthly_pi=pi,
        annual_program_fee_rate=fees.annual_fee_rate,
        monthly_program_fee=monthly_fee,
        estimated_monthly_loan_payment=float(to_cents(pi + monthly_fee)),
        total_principal=total_principal,
        total_interest=total_interest,
        first_payment_date=first_date,
        payoff_date=schedule[-1].payment_date if schedule else None,
        extra_monthly_principal=scenario.extra_monthly_principal,
        upfront_rate_source=fees.upfront_rate_source,
        annual_rate_source=fees.annual_rate_source,
        program_assumptions_verified=fees.assumptions_verified,
        notices=fees.notices,
        amortization_schedule=tuple(schedule),
    )
