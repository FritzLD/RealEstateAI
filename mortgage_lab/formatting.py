"""Display formatting helpers (no Streamlit dependency)."""

from __future__ import annotations

from datetime import date


def format_currency(value: float | None, cents: bool = True) -> str:
    """Format dollars, e.g. 1234.5 -> '$1,234.50'. Negatives use a leading minus."""
    if value is None:
        return "—"
    sign = "-" if value < 0 else ""
    pattern = "{:,.2f}" if cents else "{:,.0f}"
    return f"{sign}${pattern.format(abs(value))}"


def format_signed_currency(value: float) -> str:
    """Currency with an explicit sign for deltas, e.g. '+$125.00'."""
    if abs(value) < 0.005:
        return "$0.00"
    return ("+" if value > 0 else "") + format_currency(value)


def format_percentage(value: float | None, decimals: int = 2) -> str:
    """Format a value already in percentage points, e.g. 6.375 -> '6.375%'."""
    if value is None:
        return "—"
    return f"{value:.{decimals}f}%"


def format_rate(value: float) -> str:
    """Interest-rate style with 3 decimals, e.g. 6.375 -> '6.375%'."""
    return format_percentage(value, decimals=3)


def format_date(value: date | None) -> str:
    """Format a date as 'Nov 2026'-style month and year."""
    return value.strftime("%b %Y") if value else "—"


def format_term(term_years: int) -> str:
    """'30 Year Fixed'."""
    return f"{term_years} Year Fixed"
