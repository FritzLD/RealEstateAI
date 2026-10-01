"""Mortgage Scenario Lab — Streamlit presentation layer.

``render_mortgage_tab()`` draws the whole calculator into whatever Streamlit
container is active: the standalone app's page, or a tab inside another app
such as RealEstateAI. It never calls ``st.set_page_config``.

All calculations happen in the engine modules. This file only collects
inputs, builds a ``MortgageScenario``, and renders the structured
``MortgageResult``. It contains no program percentages and no formulas.

Every session-state and widget key is prefixed with ``KEY_PREFIX`` so the
calculator can share a page with other components without key collisions.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from mortgage_lab.formatting import (
    format_currency,
    format_date,
    format_percentage,
    format_rate,
    format_signed_currency,
    format_term,
)
from mortgage_lab.loan_programs import (
    CONVENTIONAL_PMI_NOTICE,
    UNVERIFIED_PROGRAM_NOTICE,
    ZERO_DOWN_DEFAULT_PROGRAMS,
    program_is_verified,
    schedule_rates,
    tier_table_rows,
)
from mortgage_lab.models import SCHEDULE_COLUMN_LABELS, LoanProgram, MortgageScenario, ScenarioValidationError
from mortgage_lab.mortgage_calculator import calculate_mortgage, down_payment_to_pct, pct_to_down_payment
from mortgage_lab.scenario_manager import (
    comparison_table,
    compute_deltas,
    extra_payment_impact,
    rate_sensitivity,
)
from mortgage_lab.visualizations import create_amortization_chart, create_rate_sensitivity_chart

# ---------------------------------------------------------------- copy

PAYMENT_DISCLOSURE = (
    "Estimated monthly loan payment shown here does not include property taxes, "
    "homeowner's insurance, HOA dues, Conventional PMI, or other housing expenses "
    "unless specifically shown."
)
GENERAL_DISCLOSURE = (
    "Mortgage calculations are estimates for educational and scenario-planning purposes only "
    "and are not a loan approval, commitment to lend, interest-rate quote, or rate lock. Actual "
    "loan terms, payments, mortgage insurance, guarantee fees, funding fees, eligibility, and "
    "costs depend on the borrower, property, loan program, lender pricing, and underwriting. "
    "Property taxes, homeowner's insurance, HOA dues, closing costs, and certain other costs are "
    "not included in this version of the calculator."
)

UPFRONT_FEE_LABEL = {
    LoanProgram.FHA: "Upfront MIP",
    LoanProgram.VA: "VA Funding Fee",
    LoanProgram.USDA: "Upfront Guarantee Fee",
}
MONTHLY_FEE_LABEL = {
    LoanProgram.FHA: "Monthly FHA MIP (estimate)",
    LoanProgram.USDA: "Monthly USDA Program Fee (estimate)",
}
PROGRAM_RATE_LABELS = {
    LoanProgram.FHA: {"fha_upfront_mip_rate": "Upfront MIP", "fha_annual_mip_rate": "Annual MIP"},
    LoanProgram.VA: {"va_funding_fee_rate": "Funding fee"},
    LoanProgram.USDA: {"usda_upfront_fee_rate": "Upfront guarantee fee",
                       "usda_annual_fee_rate": "Annual guarantee fee"},
}
TERM_OPTIONS = (30, 15)
SCENARIO_SLOTS = ("Scenario A", "Scenario B")

KEY_PREFIX = "msl_"


def K(name: str) -> str:
    """Namespaced session-state / widget key."""
    return KEY_PREFIX + name


class _PrefixedState:
    """``st.session_state`` view that transparently prefixes every key."""

    def __getattr__(self, name: str):
        try:
            return st.session_state[K(name)]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:
        st.session_state[K(name)] = value

    def __getitem__(self, name: str):
        return st.session_state[K(name)]

    def __setitem__(self, name: str, value) -> None:
        st.session_state[K(name)] = value

    def __contains__(self, name: str) -> bool:
        return K(name) in st.session_state

    def get(self, name: str, default=None):
        return st.session_state.get(K(name), default)

    def setdefault(self, name: str, value) -> None:
        if K(name) not in st.session_state:
            st.session_state[K(name)] = value


ss = _PrefixedState()


# ---------------------------------------------------------------- state


def init_state() -> None:
    """Seed session state once. Program rates come from loan_programs, never app.py."""
    defaults: dict = {
        "program": LoanProgram.CONVENTIONAL.value,
        "price": 400_000.0,
        "dp_amount": 40_000.0,
        "dp_pct": 10.0,
        "dp_mode": "amount",  # which down payment field the user last edited
        "rate": 6.375,
        "term": 30,
        "first_payment": None,
        "va_exempt_choice": "No",
        "extra_principal": 0.0,
        "saved": {},
    }
    for program in LoanProgram:
        defaults.update(schedule_rates(program, 0.0))
        defaults[f"finance_{program.name}"] = "Yes"
        defaults[f"override_{program.name}"] = False
    for key, value in defaults.items():
        ss.setdefault(key, value)


def current_down_pct() -> float:
    return down_payment_to_pct(ss.price, ss.dp_amount) if ss.price > 0 else 0.0


def prefill_overrides(program: LoanProgram) -> None:
    """When the user turns on overrides, start from the current schedule rates."""
    if ss.get(f"override_{program.name}"):
        for key, value in schedule_rates(program, current_down_pct()).items():
            ss[key] = value


def on_price_change() -> None:
    """Keep the controlling down payment input fixed and recompute the other."""
    if ss.price <= 0:
        return
    if ss.dp_mode == "percent" and 0 <= ss.dp_pct <= 100:
        ss.dp_amount = pct_to_down_payment(ss.price, ss.dp_pct)
    else:
        ss.dp_pct = round(down_payment_to_pct(ss.price, ss.dp_amount), 4)


def on_amount_change() -> None:
    ss.dp_mode = "amount"
    if ss.price > 0:
        ss.dp_pct = round(down_payment_to_pct(ss.price, ss.dp_amount), 4)


def on_pct_change() -> None:
    ss.dp_mode = "percent"
    if ss.price > 0 and 0 <= ss.dp_pct <= 100:
        ss.dp_amount = pct_to_down_payment(ss.price, ss.dp_pct)


def on_program_change() -> None:
    """VA and USDA default to 0% down when selected."""
    if LoanProgram.from_value(ss.program) in ZERO_DOWN_DEFAULT_PROGRAMS:
        ss.dp_amount, ss.dp_pct, ss.dp_mode = 0.0, 0.0, "percent"


# ---------------------------------------------------------------- inputs


def render_inputs() -> dict:
    """Left column. Returns keyword arguments for MortgageScenario."""
    st.subheader("Your Scenario")

    st.selectbox("Loan program", [p.value for p in LoanProgram], key=K("program"),
                 on_change=on_program_change)
    program = LoanProgram.from_value(ss.program)

    st.number_input("Purchase price / property value ($)", min_value=0.0, step=5_000.0,
                    format="%.2f", key=K("price"), on_change=on_price_change)

    c1, c2 = st.columns(2)
    c1.number_input("Down payment ($)", min_value=0.0, step=1_000.0, format="%.2f",
                    key=K("dp_amount"), on_change=on_amount_change)
    c2.number_input("Down payment (%)", min_value=0.0, step=0.5, format="%.2f",
                    key=K("dp_pct"), on_change=on_pct_change,
                    help="Dollar and percent stay in sync. Whichever you edited last is "
                         "held constant when the purchase price changes.")

    c3, c4 = st.columns(2)
    c3.number_input("Interest rate (%)", min_value=0.0, max_value=30.0, step=0.125,
                    format="%.3f", key=K("rate"),
                    help="A modeling assumption you enter. Not a quote, lender pricing, "
                         "or a locked rate.")
    c4.radio("Loan term", TERM_OPTIONS, key=K("term"), format_func=format_term, horizontal=True)

    st.date_input("First payment date (optional)", key=K("first_payment"),
                  format="MM/DD/YYYY",
                  help="If left blank, the first payment is assumed to be the 1st of next month.")

    kwargs: dict = {}
    if program is not LoanProgram.CONVENTIONAL:
        kwargs = render_program_controls(program)

    with st.expander("Advanced: extra monthly principal"):
        st.number_input("Extra principal paid each month ($)", min_value=0.0, step=50.0,
                        format="%.2f", key=K("extra_principal"),
                        help="Shows the payoff date, months saved, and interest saved. "
                             "The main summary keeps the scheduled payment.")
    return kwargs


def _rate_input(label: str, key: str, column=st) -> None:
    column.number_input(label, min_value=0.0, max_value=10.0, step=0.05, format="%.3f", key=K(key))


def render_program_controls(program: LoanProgram) -> dict:
    """Only the fields relevant to the selected government program.

    Rates follow the program schedule (including down-payment tiers) unless the
    user turns on "Override program rates".
    """
    st.markdown(f"**{program.value} program assumptions**")
    finance_key = f"finance_{program.name}"
    override_key = f"override_{program.name}"
    kwargs: dict = {}

    exempt = False
    if program is LoanProgram.VA:
        st.radio("Funding fee exempt?", ("No", "Yes"), key=K("va_exempt_choice"), horizontal=True)
        exempt = ss.va_exempt_choice == "Yes"
        kwargs["va_funding_fee_exempt"] = exempt

    if exempt:
        st.caption("Funding fee is $0 when exempt.")
    else:
        schedule = schedule_rates(program, current_down_pct())
        overriding = st.toggle("Override program rates", key=K(override_key),
                               on_change=prefill_overrides, args=(program,),
                               help="Off: rates follow the program schedule for this down payment. "
                                    "On: enter rates manually.")
        labels = PROGRAM_RATE_LABELS[program]
        if overriding:
            cols = st.columns(len(schedule))
            for col, key in zip(cols, schedule):
                _rate_input(f"{labels[key]} (%)", key, col)
                kwargs[key] = ss[key]
        else:
            st.caption("Schedule rates at " + format_percentage(current_down_pct()) + " down: "
                       + " · ".join(f"{labels[k]} {format_percentage(v, 3)}"
                                    for k, v in schedule.items()))
            # None tells the engine to use the program schedule.
            kwargs.update({key: None for key in schedule})
        with st.expander("Program rate schedule"):
            st.dataframe(pd.DataFrame(tier_table_rows(program)), hide_index=True, width="stretch")
        finance_label = {LoanProgram.FHA: "Finance upfront MIP?",
                         LoanProgram.VA: "Finance funding fee?",
                         LoanProgram.USDA: "Finance upfront fee?"}[program]
        st.radio(finance_label, ("Yes", "No"), key=K(finance_key), horizontal=True)

    if program is LoanProgram.VA:
        st.caption("VA loans do not carry monthly mortgage insurance.")

    kwargs["finance_program_fee"] = ss.get(finance_key, "Yes") == "Yes"
    if not program_is_verified(program):
        st.info(UNVERIFIED_PROGRAM_NOTICE, icon=":material/info:")
    return kwargs


def build_scenario(program_kwargs: dict) -> MortgageScenario:
    return MortgageScenario(
        loan_program=ss.program,
        purchase_price=ss.price,
        down_payment=ss.dp_amount,
        interest_rate=ss.rate,
        term_years=int(ss.term),
        first_payment_date=ss.first_payment,
        extra_monthly_principal=0.0,
        **program_kwargs,
    )


# ---------------------------------------------------------------- summary


def _source_suffix(source: str) -> str:
    return f", {source.lower()}" if source else ""


def summary_rows(result, program: LoanProgram) -> list[tuple[str, str]]:
    rows = [
        ("Purchase Price", format_currency(result.purchase_price)),
        ("Down Payment", format_currency(result.down_payment)),
        ("Down Payment %", format_percentage(result.down_payment_pct)),
        ("Base Loan Amount", format_currency(result.base_loan_amount)),
    ]
    if program in UPFRONT_FEE_LABEL:
        label = UPFRONT_FEE_LABEL[program]
        rows += [
            (f"{label} ({format_percentage(result.upfront_fee_rate, 3)}"
             f"{_source_suffix(result.upfront_rate_source)})",
             format_currency(result.upfront_program_fee)),
            (f"{label} — financed", format_currency(result.financed_program_fee)),
            (f"{label} — paid separately", format_currency(result.upfront_fee_paid_separately)),
        ]
    rows += [
        ("Total Financed Loan Amount", format_currency(result.total_financed_loan_amount)),
        ("Interest Rate", format_rate(result.interest_rate)),
        ("Loan Term", format_term(result.term_years)),
        ("Monthly Principal & Interest", format_currency(result.monthly_pi)),
    ]
    if program in MONTHLY_FEE_LABEL:
        rows.append((f"{MONTHLY_FEE_LABEL[program]} "
                     f"({format_percentage(result.annual_program_fee_rate, 3)} annual"
                     f"{_source_suffix(result.annual_rate_source)})",
                     format_currency(result.monthly_program_fee)))
    rows += [
        ("Estimated Monthly Loan Payment", format_currency(result.estimated_monthly_loan_payment)),
        ("Total Scheduled Mortgage Interest", format_currency(result.total_interest)),
        ("First Payment", format_date(result.first_payment_date)),
        ("Payoff Date", format_date(result.payoff_date)),
    ]
    return rows


def render_summary(scenario: MortgageScenario, result) -> None:
    program = scenario.loan_program
    st.subheader("Mortgage Summary")

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated monthly loan payment", format_currency(result.estimated_monthly_loan_payment))
    m2.metric("Monthly P&I", format_currency(result.monthly_pi))
    m3.metric("Total financed loan", format_currency(result.total_financed_loan_amount))
    st.caption(PAYMENT_DISCLOSURE)

    for notice in result.notices:
        if notice == CONVENTIONAL_PMI_NOTICE:
            st.warning(notice, icon=":material/info:")
        elif notice != UNVERIFIED_PROGRAM_NOTICE:  # unverified notice is shown with the inputs
            st.info(notice, icon=":material/info:")

    table = pd.DataFrame(summary_rows(result, program), columns=["Item", "Amount"])
    st.dataframe(table, hide_index=True, width="stretch", height=38 + 35 * len(table))

    if ss.extra_principal > 0 and result.base_loan_amount > 0:
        impact = extra_payment_impact(
            replace(scenario, first_payment_date=result.first_payment_date,
                    extra_monthly_principal=ss.extra_principal)
        )
        st.markdown(f"**With {format_currency(impact.extra_monthly_principal)} extra principal "
                    "each month**")
        e1, e2, e3 = st.columns(3)
        e1.metric("New payoff date", format_date(impact.new_payoff_date))
        e2.metric("Months saved", f"{impact.months_saved}")
        e3.metric("Interest saved", format_currency(impact.interest_saved))


# ---------------------------------------------------------------- lower sections


def render_amortization(result) -> None:
    if not result.amortization_schedule:
        st.info("There is no loan balance to amortize for this scenario.")
        return
    st.plotly_chart(create_amortization_chart(result), width="stretch", key=K("amort_chart"))
    st.caption("Program fees (FHA MIP, USDA annual fee) are not principal or interest and are "
               "not included in these lines.")

    with st.expander("View Full Amortization Schedule"):
        df = result.schedule_dataframe().drop(columns=["extra_principal"])
        df = df.rename(columns=SCHEDULE_COLUMN_LABELS)
        money = {c: st.column_config.NumberColumn(c, format="dollar")
                 for c in df.columns if c not in ("Payment #", "Payment Date")}
        st.dataframe(df, hide_index=True, width="stretch", height=420,
                     column_config={"Payment Date": st.column_config.DateColumn(format="MMM YYYY"),
                                    **money})
        st.download_button("Download schedule (CSV)", df.to_csv(index=False).encode(),
                           file_name="amortization_schedule.csv", mime="text/csv",
                           key=K("download_schedule"))


def render_comparison(result) -> None:
    st.caption("Save the current assumptions, change inputs above, and compare. "
               "Scenarios stay in this browser session only.")
    b1, b2, b3, _ = st.columns([1, 1, 1, 2])
    if b1.button("Save as Scenario A", key=K("save_a"), width="stretch"):
        ss.saved["Scenario A"] = result
    if b2.button("Save as Scenario B", key=K("save_b"), width="stretch"):
        ss.saved["Scenario B"] = result
    if b3.button("Clear saved", key=K("clear_saved"), width="stretch", disabled=not ss.saved):
        ss.saved = {}

    saved = {slot: ss.saved[slot] for slot in SCENARIO_SLOTS if slot in ss.saved}
    if not saved:
        st.info("No saved scenarios yet.")
        return

    st.dataframe(comparison_table({**saved, "Current": result}), width="stretch")

    st.markdown("**Differences (Current minus saved scenario)**")
    delta_rows = []
    for slot, saved_result in saved.items():
        d = compute_deltas(saved_result, result)
        delta_rows.append({
            "Compared with": slot,
            "Monthly Payment Difference": format_signed_currency(d.monthly_payment_difference),
            "Down Payment Difference": format_signed_currency(d.down_payment_difference),
            "Financed Loan Difference": format_signed_currency(d.financed_loan_difference),
            "Total Interest Difference": format_signed_currency(d.total_interest_difference),
        })
    st.dataframe(pd.DataFrame(delta_rows), hide_index=True, width="stretch")
    st.caption("A positive value means the current scenario's figure is higher. "
               "This is a factual comparison, not a recommendation.")


def render_sensitivity(scenario: MortgageScenario) -> None:
    table = rate_sensitivity(scenario)
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        shown = pd.DataFrame({
            "Interest Rate": table["interest_rate"].map(format_rate),
            "Change": table["rate_offset"].map(lambda o: "Entered rate" if o == 0 else f"{o:+.2f}%"),
            "Monthly P&I": table["monthly_pi"].map(format_currency),
            "vs. Entered": table["pi_change_vs_current"].map(format_signed_currency),
        })
        st.dataframe(shown, hide_index=True, width="stretch")
        st.caption("All other assumptions held constant. Rates below 0% are skipped.")
    with right:
        st.plotly_chart(create_rate_sensitivity_chart(table, scenario.interest_rate),
                        width="stretch", key=K("rate_chart"))


# ---------------------------------------------------------------- page


DEFAULT_SUBTITLE = "Model and compare mortgage financing scenarios."


def render_mortgage_tab(title: str | None = None, subtitle: str | None = DEFAULT_SUBTITLE) -> None:
    """Render the full calculator into the current Streamlit container.

    Args:
        title: Optional header drawn with ``st.header`` (e.g. inside a host tab).
        subtitle: Optional caption under the title; pass ``None`` to omit.
    """
    init_state()

    if title:
        st.header(title)
    if subtitle:
        st.caption(subtitle)

    left, right = st.columns([1, 1.3], gap="large")
    with left:
        program_kwargs = render_inputs()

    try:
        scenario = build_scenario(program_kwargs)
        result = calculate_mortgage(scenario)
    except ScenarioValidationError as exc:
        with right:
            st.subheader("Mortgage Summary")
            st.error(str(exc), icon=":material/error:")
        st.caption(GENERAL_DISCLOSURE)
        return

    with right:
        render_summary(scenario, result)

    st.divider()
    tab_amort, tab_compare, tab_rate = st.tabs(
        ["Amortization", "Scenario Comparison", "Rate Sensitivity"]
    )
    with tab_amort:
        render_amortization(result)
    with tab_compare:
        render_comparison(result)
    with tab_rate:
        render_sensitivity(scenario)

    st.divider()
    st.caption(GENERAL_DISCLOSURE)
