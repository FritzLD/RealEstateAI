"""Government loan program assumptions and fee rules.

THIS MODULE IS THE SINGLE SOURCE OF TRUTH for every program percentage the
application uses. No other module contains FHA, VA, or USDA rates.

Verification status
-------------------
The rates and tiers below were reviewed by the project owner on 2026-10-01
and confirmed as current, within the scope noted on each config. Anything
outside that scope (for example, FHA terms of 15 years or less, or VA
subsequent use) is NOT modelled and is flagged in the result notices so the
user can enter a rate manually.

How tiers work
--------------
Rates that depend on down payment are stored as ``RateTier`` tables. Each
tier applies from its ``min_down_payment_pct`` (inclusive) up to the next
tier. Example for the VA funding fee (first use):

    0%  <= down < 5%   -> 2.15%
    5%  <= down < 10%  -> 1.50%
    10% <= down        -> 1.25%

Down payment percentage is used as a proxy for LTV (base loan / purchase
price). Agencies compute LTV against the lesser of price or appraised value;
this calculator assumes they are equal.

Known simplifications
---------------------
* Annual fees (FHA annual MIP, USDA annual fee) are estimated as
  ``base loan amount x annual rate / 12`` and held flat for every payment.
  Agencies compute these on a declining/average balance, and FHA MIP may
  end before the loan term does.
* FHA annual MIP tiers apply to terms longer than 15 years at standard loan
  amounts. Higher-balance loan amounts can carry a different annual MIP and
  are not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable

from mortgage_lab.models import LoanProgram, MortgageScenario

VERIFIED_ON = "2026-10-01"

UNVERIFIED_PROGRAM_NOTICE = (
    "Government loan program assumptions shown in this prototype must be verified "
    "against current agency guidance before production use."
)

CONVENTIONAL_PMI_NOTICE = (
    "Conventional mortgage insurance may apply. "
    "PMI is not included in this version of the calculator."
)

FHA_SHORT_TERM_NOTICE = (
    "FHA annual MIP for terms of 15 years or less follows a different schedule that is not "
    "modeled here. The rate shown uses the longer-term schedule; override it with the "
    "applicable rate."
)

VA_SUBSEQUENT_USE_NOTICE = (
    "VA funding fee rates shown are first-use rates. Subsequent-use rates differ; "
    "override the rate if this is not a first use."
)

# Down payment below which the Conventional PMI notice is shown (percent of price).
# This only controls a disclosure; no PMI is calculated.
CONVENTIONAL_PMI_NOTICE_THRESHOLD_PCT = 20.0

# Programs whose UI default down payment is $0 when the program is selected.
ZERO_DOWN_DEFAULT_PROGRAMS: frozenset[LoanProgram] = frozenset({LoanProgram.VA, LoanProgram.USDA})

RATE_SOURCE_SCHEDULE = "Program schedule"
RATE_SOURCE_OVERRIDE = "User override"
RATE_SOURCE_EXEMPT = "Exempt"

_CENT = Decimal("0.01")


# --------------------------------------------------------------------------
# Tier tables
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RateTier:
    """A rate that applies from ``min_down_payment_pct`` (inclusive) upward."""

    min_down_payment_pct: float
    rate_pct: float


def rate_for_down_payment(tiers: tuple[RateTier, ...], down_payment_pct: float) -> float:
    """Return the rate of the highest tier whose minimum the down payment meets.

    The down payment percentage is rounded to 6 decimals first so values such as
    4.9999999 (floating-point noise from a 5% entry) land in the intended tier.
    """
    if not tiers:
        raise ValueError("Tier table is empty.")
    ordered = sorted(tiers, key=lambda t: t.min_down_payment_pct)
    pct = round(down_payment_pct, 6)
    rate = ordered[0].rate_pct
    for tier in ordered:
        if pct >= tier.min_down_payment_pct:
            rate = tier.rate_pct
    return rate


# --------------------------------------------------------------------------
# Program configuration (audit and update here)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FHAProgramConfig:
    """FHA mortgage insurance assumptions (percentages)."""

    upfront_mip_pct: float
    annual_mip_tiers: tuple[RateTier, ...]
    # Annual MIP tiers apply to terms longer than this many years.
    annual_tiers_min_term_years_exclusive: int = 15
    verified: bool = True
    source_note: str = f"Reviewed by project owner {VERIFIED_ON}"

    def annual_mip_pct(self, down_payment_pct: float) -> float:
        return rate_for_down_payment(self.annual_mip_tiers, down_payment_pct)


@dataclass(frozen=True)
class VAProgramConfig:
    """VA funding fee assumptions (percentages, first use). VA has no monthly MI."""

    funding_fee_tiers: tuple[RateTier, ...]
    verified: bool = True
    source_note: str = f"Reviewed by project owner {VERIFIED_ON} (first use)"

    def funding_fee_pct(self, down_payment_pct: float) -> float:
        return rate_for_down_payment(self.funding_fee_tiers, down_payment_pct)


@dataclass(frozen=True)
class USDAProgramConfig:
    """USDA guarantee fee assumptions (percentages)."""

    upfront_guarantee_fee_pct: float
    annual_guarantee_fee_pct: float
    verified: bool = True
    source_note: str = f"Reviewed by project owner {VERIFIED_ON}"


# FHA — upfront MIP flat; annual MIP by down payment (terms > 15 years).
#   down payment < 5%  (LTV > 95%)  -> 0.55%
#   down payment >= 5% (LTV <= 95%) -> 0.50%
FHA_CONFIG = FHAProgramConfig(
    upfront_mip_pct=1.75,
    annual_mip_tiers=(
        RateTier(min_down_payment_pct=0.0, rate_pct=0.55),
        RateTier(min_down_payment_pct=5.0, rate_pct=0.50),
    ),
)

# VA — funding fee by down payment (first use).
#   down payment < 5%        -> 2.15%
#   5% <= down payment < 10% -> 1.50%
#   down payment >= 10%      -> 1.25%
VA_CONFIG = VAProgramConfig(
    funding_fee_tiers=(
        RateTier(min_down_payment_pct=0.0, rate_pct=2.15),
        RateTier(min_down_payment_pct=5.0, rate_pct=1.50),
        RateTier(min_down_payment_pct=10.0, rate_pct=1.25),
    ),
)

# USDA — flat upfront and annual guarantee fees.
USDA_CONFIG = USDAProgramConfig(
    upfront_guarantee_fee_pct=1.00,
    annual_guarantee_fee_pct=0.35,
)


# --------------------------------------------------------------------------
# Fee calculation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgramFees:
    """Program fee breakdown for one scenario (currency rounded to cents)."""

    upfront_fee_rate: float = 0.0
    upfront_fee: float = 0.0
    financed_upfront_fee: float = 0.0
    upfront_fee_paid_separately: float = 0.0
    annual_fee_rate: float = 0.0
    monthly_program_fee: float = 0.0
    upfront_rate_source: str = ""
    annual_rate_source: str = ""
    assumptions_verified: bool = True
    notices: tuple[str, ...] = field(default=())


def _cents(value: float) -> float:
    """Round a currency amount to cents, half-up."""
    return float(Decimal(str(value)).quantize(_CENT, rounding=ROUND_HALF_UP))


def _upfront_split(base_loan: float, rate_pct: float, finance: bool) -> tuple[float, float, float]:
    """Return (upfront fee, financed portion, paid-separately portion)."""
    fee = _cents(base_loan * rate_pct / 100)
    return (fee, fee, 0.0) if finance else (fee, 0.0, fee)


def _monthly_from_annual(base_loan: float, annual_rate_pct: float) -> float:
    """Flat monthly estimate of an annual fee charged on the base loan amount."""
    return _cents(base_loan * annual_rate_pct / 100 / 12)


def _resolve(override: float | None, schedule_rate: float) -> tuple[float, str]:
    """Use the override when supplied, otherwise the program schedule."""
    if override is None:
        return schedule_rate, RATE_SOURCE_SCHEDULE
    return float(override), RATE_SOURCE_OVERRIDE


def down_payment_pct_of(scenario: MortgageScenario) -> float:
    """Down payment as a percentage of purchase price (full precision)."""
    return scenario.down_payment / scenario.purchase_price * 100


def conventional_fees(scenario: MortgageScenario, base_loan: float) -> ProgramFees:
    """Conventional: no government program fee. PMI is not calculated."""
    notices: tuple[str, ...] = ()
    if base_loan > 0 and down_payment_pct_of(scenario) < CONVENTIONAL_PMI_NOTICE_THRESHOLD_PCT:
        notices = (CONVENTIONAL_PMI_NOTICE,)
    return ProgramFees(assumptions_verified=True, notices=notices)


def fha_fees(
    scenario: MortgageScenario, base_loan: float, config: FHAProgramConfig = FHA_CONFIG
) -> ProgramFees:
    """FHA: upfront MIP (optionally financed) plus monthly MIP from the down payment tier."""
    down_pct = down_payment_pct_of(scenario)
    upfront_rate, upfront_src = _resolve(scenario.fha_upfront_mip_rate, config.upfront_mip_pct)
    annual_rate, annual_src = _resolve(scenario.fha_annual_mip_rate, config.annual_mip_pct(down_pct))

    in_scope = scenario.term_years > config.annual_tiers_min_term_years_exclusive
    verified = config.verified and (in_scope or annual_src == RATE_SOURCE_OVERRIDE)
    notices: list[str] = []
    if not in_scope and annual_src == RATE_SOURCE_SCHEDULE:
        notices.append(FHA_SHORT_TERM_NOTICE)
    if not config.verified:
        notices.append(UNVERIFIED_PROGRAM_NOTICE)

    fee, financed, separate = _upfront_split(base_loan, upfront_rate, scenario.finance_program_fee)
    return ProgramFees(
        upfront_fee_rate=upfront_rate,
        upfront_fee=fee,
        financed_upfront_fee=financed,
        upfront_fee_paid_separately=separate,
        annual_fee_rate=annual_rate,
        monthly_program_fee=_monthly_from_annual(base_loan, annual_rate),
        upfront_rate_source=upfront_src,
        annual_rate_source=annual_src,
        assumptions_verified=verified,
        notices=tuple(notices),
    )


def va_fees(
    scenario: MortgageScenario, base_loan: float, config: VAProgramConfig = VA_CONFIG
) -> ProgramFees:
    """VA: funding fee from the down payment tier (optionally financed, $0 if exempt)."""
    notices: list[str] = []
    if scenario.va_funding_fee_exempt:
        rate, source = 0.0, RATE_SOURCE_EXEMPT
    else:
        rate, source = _resolve(
            scenario.va_funding_fee_rate, config.funding_fee_pct(down_payment_pct_of(scenario))
        )
        if source == RATE_SOURCE_SCHEDULE:
            notices.append(VA_SUBSEQUENT_USE_NOTICE)
    if not config.verified:
        notices.append(UNVERIFIED_PROGRAM_NOTICE)

    fee, financed, separate = _upfront_split(base_loan, rate, scenario.finance_program_fee)
    return ProgramFees(
        upfront_fee_rate=rate,
        upfront_fee=fee,
        financed_upfront_fee=financed,
        upfront_fee_paid_separately=separate,
        annual_fee_rate=0.0,
        monthly_program_fee=0.0,
        upfront_rate_source=source,
        annual_rate_source="",
        assumptions_verified=config.verified,
        notices=tuple(notices),
    )


def usda_fees(
    scenario: MortgageScenario, base_loan: float, config: USDAProgramConfig = USDA_CONFIG
) -> ProgramFees:
    """USDA: upfront guarantee fee (optionally financed) plus monthly annual-fee estimate."""
    upfront_rate, upfront_src = _resolve(scenario.usda_upfront_fee_rate,
                                         config.upfront_guarantee_fee_pct)
    annual_rate, annual_src = _resolve(scenario.usda_annual_fee_rate,
                                       config.annual_guarantee_fee_pct)
    fee, financed, separate = _upfront_split(base_loan, upfront_rate, scenario.finance_program_fee)
    return ProgramFees(
        upfront_fee_rate=upfront_rate,
        upfront_fee=fee,
        financed_upfront_fee=financed,
        upfront_fee_paid_separately=separate,
        annual_fee_rate=annual_rate,
        monthly_program_fee=_monthly_from_annual(base_loan, annual_rate),
        upfront_rate_source=upfront_src,
        annual_rate_source=annual_src,
        assumptions_verified=config.verified,
        notices=() if config.verified else (UNVERIFIED_PROGRAM_NOTICE,),
    )


PROGRAM_FEE_CALCULATORS: dict[LoanProgram, Callable[[MortgageScenario, float], ProgramFees]] = {
    LoanProgram.CONVENTIONAL: conventional_fees,
    LoanProgram.FHA: fha_fees,
    LoanProgram.VA: va_fees,
    LoanProgram.USDA: usda_fees,
}


def calculate_program_fees(scenario: MortgageScenario, base_loan: float) -> ProgramFees:
    """Dispatch to the fee rule for the scenario's program."""
    return PROGRAM_FEE_CALCULATORS[scenario.loan_program](scenario, base_loan)


def schedule_rates(program: LoanProgram, down_payment_pct: float) -> dict[str, float]:
    """Rates the program schedule gives for a down payment, keyed by scenario field name.

    The UI uses this to display schedule rates and to pre-fill override inputs,
    so no percentage lives in app.py.
    """
    if program is LoanProgram.FHA:
        return {
            "fha_upfront_mip_rate": FHA_CONFIG.upfront_mip_pct,
            "fha_annual_mip_rate": FHA_CONFIG.annual_mip_pct(down_payment_pct),
        }
    if program is LoanProgram.VA:
        return {"va_funding_fee_rate": VA_CONFIG.funding_fee_pct(down_payment_pct)}
    if program is LoanProgram.USDA:
        return {
            "usda_upfront_fee_rate": USDA_CONFIG.upfront_guarantee_fee_pct,
            "usda_annual_fee_rate": USDA_CONFIG.annual_guarantee_fee_pct,
        }
    return {}


def tier_table_rows(program: LoanProgram) -> list[dict[str, str]]:
    """Human-readable tier table for display (e.g. in a help expander)."""

    def rows(tiers: tuple[RateTier, ...], label: str) -> list[dict[str, str]]:
        ordered = sorted(tiers, key=lambda t: t.min_down_payment_pct)
        out = []
        for i, tier in enumerate(ordered):
            upper = ordered[i + 1].min_down_payment_pct if i + 1 < len(ordered) else None
            band = (f"{tier.min_down_payment_pct:g}% to under {upper:g}%" if upper is not None
                    else f"{tier.min_down_payment_pct:g}% or more")
            out.append({"Fee": label, "Down payment": band, "Rate": f"{tier.rate_pct:.2f}%"})
        return out

    if program is LoanProgram.FHA:
        return ([{"Fee": "Upfront MIP", "Down payment": "Any",
                  "Rate": f"{FHA_CONFIG.upfront_mip_pct:.2f}%"}]
                + rows(FHA_CONFIG.annual_mip_tiers, "Annual MIP (terms over 15 years)"))
    if program is LoanProgram.VA:
        return rows(VA_CONFIG.funding_fee_tiers, "Funding fee (first use)")
    if program is LoanProgram.USDA:
        return [
            {"Fee": "Upfront guarantee fee", "Down payment": "Any",
             "Rate": f"{USDA_CONFIG.upfront_guarantee_fee_pct:.2f}%"},
            {"Fee": "Annual guarantee fee", "Down payment": "Any",
             "Rate": f"{USDA_CONFIG.annual_guarantee_fee_pct:.2f}%"},
        ]
    return []


def program_is_verified(program: LoanProgram) -> bool:
    """True when the program's configured assumptions have been reviewed."""
    configs = {LoanProgram.FHA: FHA_CONFIG, LoanProgram.VA: VA_CONFIG, LoanProgram.USDA: USDA_CONFIG}
    config = configs.get(program)
    return True if config is None else config.verified
