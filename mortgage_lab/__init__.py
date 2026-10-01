"""Mortgage Scenario Lab calculation engine.

Everything in ``mortgage_lab`` except ``ui.py`` is free of Streamlit,
so the engine can be imported and called from any Python process (tests, a
notebook, or a future RealEstateAI tool).

Typical use::

    from mortgage_lab import MortgageScenario, calculate_mortgage

    result = calculate_mortgage(
        MortgageScenario(
            loan_program="Conventional",
            purchase_price=400_000,
            down_payment=20_000,
            interest_rate=6.375,
            term_years=30,
        )
    )
    result.to_dict()
"""

from mortgage_lab.models import (
    AmortizationRow,
    LoanProgram,
    MortgageResult,
    MortgageScenario,
    ScenarioValidationError,
)
from mortgage_lab.mortgage_calculator import calculate_mortgage

__all__ = [
    "AmortizationRow",
    "LoanProgram",
    "MortgageResult",
    "MortgageScenario",
    "ScenarioValidationError",
    "calculate_mortgage",
]
