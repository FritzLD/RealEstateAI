"""Monthly amortization schedule and payment dates.

Rounding strategy (applied consistently):

1. The scheduled P&I payment is computed from the standard formula at full
   floating-point precision, then rounded half-up to cents, because a real
   payment is charged in cents.
2. Each month, interest = beginning balance x monthly rate, rounded half-up
   to cents. Principal = scheduled payment - interest (plus any extra
   principal). Balances are tracked in ``Decimal`` cents, so no float drift
   accumulates over 360 rows.
3. The final payment (or any payment where principal would exceed the
   remaining balance) pays off exactly the remaining balance. The ending
   balance is therefore exactly $0.00, never negative, and cumulative
   principal equals the financed principal to the cent.

Only the financed principal is amortized. A financed upfront program fee is
already part of that principal. Recurring program fees (FHA MIP, USDA annual
fee) are carried in a separate column and never touch principal, interest,
or balance.
"""

from __future__ import annotations

import calendar
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from mortgage_lab.models import AmortizationRow

_CENT = Decimal("0.01")
_ZERO = Decimal("0.00")
MONTHS_PER_YEAR = 12


def to_cents(value: float | Decimal) -> Decimal:
    """Convert to ``Decimal`` rounded half-up to cents."""
    return Decimal(str(value)).quantize(_CENT, rounding=ROUND_HALF_UP)


def add_months(start: date, months: int) -> date:
    """Add whole months to a date, clamping the day to the target month's length."""
    month_index = start.month - 1 + months
    year = start.year + month_index // MONTHS_PER_YEAR
    month = month_index % MONTHS_PER_YEAR + 1
    day = min(start.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def default_first_payment_date(today: date | None = None) -> date:
    """Prototype default: the first day of the month after ``today``.

    Real first payment dates depend on the closing date; this is a documented
    simplification for scenario planning.
    """
    today = today or date.today()
    return add_months(date(today.year, today.month, 1), 1)


def scheduled_payment_date(first_payment_date: date, number_of_payments: int) -> date:
    """Date of the last scheduled payment: first date + (n - 1) months."""
    return add_months(first_payment_date, number_of_payments - 1)


def build_schedule(
    principal: float,
    annual_rate_pct: float,
    number_of_payments: int,
    scheduled_pi: float,
    first_payment_date: date,
    monthly_program_fee: float = 0.0,
    extra_monthly_principal: float = 0.0,
) -> list[AmortizationRow]:
    """Build the month-by-month schedule.

    Args:
        principal: Financed principal (base loan + any financed upfront fee).
        annual_rate_pct: Annual note rate as a percentage.
        number_of_payments: Scheduled number of monthly payments.
        scheduled_pi: Scheduled monthly P&I, already rounded to cents.
        first_payment_date: Date of payment 1.
        monthly_program_fee: Recurring program fee shown separately each month.
        extra_monthly_principal: Optional extra principal paid each month.

    Returns:
        One :class:`AmortizationRow` per payment. With extra principal the loan
        may pay off before ``number_of_payments``.
    """
    balance = to_cents(principal)
    if balance <= _ZERO:
        return []

    monthly_rate = Decimal(str(annual_rate_pct)) / Decimal(100) / Decimal(MONTHS_PER_YEAR)
    payment = max(to_cents(scheduled_pi), _CENT)
    extra = to_cents(extra_monthly_principal)
    fee = to_cents(monthly_program_fee)

    rows: list[AmortizationRow] = []
    cumulative_principal = _ZERO
    cumulative_interest = _ZERO

    for number in range(1, number_of_payments + 1):
        beginning = balance
        interest = (beginning * monthly_rate).quantize(_CENT, rounding=ROUND_HALF_UP)
        scheduled_principal = max(payment - interest, _ZERO)
        is_final_scheduled = number == number_of_payments

        if is_final_scheduled or scheduled_principal >= beginning:
            # Pay off exactly what remains; the payment adjusts by the rounding residue.
            principal_paid = beginning
            extra_paid = _ZERO
            pi_paid = beginning + interest
        else:
            extra_paid = min(extra, beginning - scheduled_principal)
            principal_paid = scheduled_principal + extra_paid
            pi_paid = payment

        balance = beginning - principal_paid
        cumulative_principal += principal_paid
        cumulative_interest += interest

        rows.append(
            AmortizationRow(
                payment_number=number,
                payment_date=add_months(first_payment_date, number - 1),
                beginning_balance=float(beginning),
                scheduled_pi=float(pi_paid),
                principal=float(principal_paid),
                interest=float(interest),
                extra_principal=float(extra_paid),
                program_fee=float(fee),
                total_payment=float(pi_paid + extra_paid + fee),
                ending_balance=float(balance),
                cumulative_principal=float(cumulative_principal),
                cumulative_interest=float(cumulative_interest),
            )
        )
        if balance <= _ZERO:
            break

    return rows
