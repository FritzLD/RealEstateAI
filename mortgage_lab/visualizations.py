"""Plotly figures. Returns ``go.Figure`` objects; no Streamlit dependency."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from mortgage_lab.models import MortgageResult

# Restrained palette: navy, teal, slate. Interest is not red, fees never appear.
COLOR_BALANCE = "#1F3A5F"
COLOR_PRINCIPAL = "#2A9D8F"
COLOR_INTEREST = "#8D99AE"
COLOR_CURRENT = "#1F3A5F"
FONT_FAMILY = "Inter, Segoe UI, Helvetica, Arial, sans-serif"


def _base_layout(fig: go.Figure, title: str, y_title: str, x_title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=16)),
        font=dict(family=FONT_FAMILY, size=13),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(title=x_title, showgrid=False, zeroline=False)
    fig.update_yaxes(title=y_title, tickprefix="$", tickformat=",.0f",
                     gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    return fig


def create_amortization_chart(result: MortgageResult) -> go.Figure:
    """Remaining balance, cumulative principal, and cumulative interest by year.

    Calculations are monthly; the x-axis shows years (payment number / 12).
    Program fees are excluded from every line.
    """
    df = result.schedule_dataframe()
    fig = go.Figure()
    if df.empty:
        return _base_layout(fig, "Loan Balance Over Time", "Dollars", "Years")

    years = df["payment_number"] / 12
    dates = pd.to_datetime(df["payment_date"]).dt.strftime("%b %Y")
    series = (
        ("Remaining Principal Balance", "ending_balance", COLOR_BALANCE),
        ("Cumulative Principal Paid", "cumulative_principal", COLOR_PRINCIPAL),
        ("Cumulative Interest Paid", "cumulative_interest", COLOR_INTEREST),
    )
    for name, column, color in series:
        fig.add_trace(
            go.Scatter(
                x=years,
                y=df[column],
                name=name,
                mode="lines",
                line=dict(color=color, width=2.5),
                customdata=dates,
                hovertemplate="%{customdata}: $%{y:,.2f}<extra>" + name + "</extra>",
            )
        )
    fig = _base_layout(fig, "Loan Balance Over Time", "Dollars", "Years")
    fig.update_xaxes(dtick=5 if result.term_years > 15 else 1, range=[0, years.max()])
    return fig


def create_rate_sensitivity_chart(sensitivity: pd.DataFrame, current_rate: float) -> go.Figure:
    """Monthly P&I versus interest rate, with the current rate highlighted."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sensitivity["interest_rate"],
            y=sensitivity["monthly_pi"],
            mode="lines+markers",
            name="Monthly P&I",
            line=dict(color=COLOR_PRINCIPAL, width=2.5),
            marker=dict(size=8),
            hovertemplate="%{x:.3f}%: $%{y:,.2f}<extra>Monthly P&I</extra>",
        )
    )
    current = sensitivity[sensitivity["rate_offset"] == 0]
    if not current.empty:
        fig.add_trace(
            go.Scatter(
                x=current["interest_rate"],
                y=current["monthly_pi"],
                mode="markers",
                name="Entered rate",
                marker=dict(size=13, color=COLOR_CURRENT, symbol="diamond"),
                hovertemplate="%{x:.3f}%: $%{y:,.2f}<extra>Entered rate</extra>",
            )
        )
    fig = _base_layout(fig, "Monthly P&I by Interest Rate", "Monthly P&I", "Interest rate (%)")
    fig.update_xaxes(ticksuffix="%", tickformat=".3f")
    fig.update_layout(hovermode="closest", height=340)
    return fig
