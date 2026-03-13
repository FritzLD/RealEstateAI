"""
RealEstateAI – Streamlit Application
Run: streamlit run app.py

Tab 1 – Market Dashboard  : live KPIs + key charts
Tab 2 – AI Agent Chat     : conversational real estate analyst
Tab 3 – Forecasts & Analysis : 12-month outlook, refi windows, model evaluation
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from src import config
from src.data_loader import RealEstateDataLoader
from src.forecasting import MarketForecaster
from src.knowledge_base import KnowledgeBase
from src.llm_chain import RealEstateChain
from src.refi_analysis import RefiAnalyzer
from src.retriever import RealEstateRetriever
from src.visualizations import RealEstateVisualizer

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── System initialisation (cached) ───────────────────────────────────────────

@st.cache_resource(show_spinner="Loading market data and building knowledge base…")
def build_system(api_key: str, model_name: str) -> dict:
    """One-time initialisation: load data, build vector store, wire agent."""
    loader     = RealEstateDataLoader()
    forecaster = MarketForecaster(loader.df, loader.exog_cols)
    refi       = RefiAnalyzer(loader.df)
    visualizer = RealEstateVisualizer(loader.df)
    summary    = loader.get_market_summary()

    # RAG knowledge base
    documents = loader.generate_knowledge_documents()
    kb        = KnowledgeBase(api_key=api_key)
    kb.get_or_create(documents)
    base_retriever = kb.get_retriever(k=config.TOP_K_RETRIEVAL)

    # Hybrid retriever
    hybrid_retriever = RealEstateRetriever(
        vector_retriever=base_retriever,
        loader=loader,
        forecaster=forecaster,
    )

    # LLM chain
    chain = RealEstateChain(
        api_key=api_key,
        model_name=model_name,
        retriever=hybrid_retriever,
    )

    return {
        "loader":     loader,
        "forecaster": forecaster,
        "refi":       refi,
        "visualizer": visualizer,
        "chain":      chain,
        "summary":    summary,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, str, bool]:
    st.sidebar.title(f"{config.APP_ICON} RealEstateAI")
    st.sidebar.markdown("**Dayton MSA Market Intelligence**")
    st.sidebar.divider()

    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=config.OPENAI_API_KEY,
        type="password",
        help="Your OpenAI key.  Set OPENAI_API_KEY in .env to avoid entering it here.",
    )
    model = st.sidebar.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        index=0,
    )

    init_clicked = st.sidebar.button("Initialize System", type="primary", use_container_width=True)

    if "system" in st.session_state:
        s = st.session_state.system["summary"]
        st.sidebar.divider()
        st.sidebar.markdown("**Market Snapshot**")
        st.sidebar.metric("Data Through",    s["data_through"])
        st.sidebar.metric("Active Listings", f"{s['current_active']:,}")
        st.sidebar.metric("Monthly Sales",   f"{s['current_sales']:,}")
        st.sidebar.metric("Disparity %",     f"{s['disparity_pct']}%")
        st.sidebar.metric("30-Yr Rate",      f"{s['current_rate']}%")
        st.sidebar.metric("YoY Sales Δ",     f"{s['yoy_sales_chg']:+}%")

    return api_key, model, init_clicked


# ── Tab 1: Market Dashboard ───────────────────────────────────────────────────

def render_dashboard(sys: dict) -> None:
    st.header("Market Dashboard")

    s = sys["summary"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active Listings",    f"{s['current_active']:,}",  f"{s['yoy_active_chg']:+}% YoY")
    c2.metric("Monthly Sales",      f"{s['current_sales']:,}",   f"{s['yoy_sales_chg']:+}% YoY")
    c3.metric("Market Disparity %", f"{s['disparity_pct']}%")
    c4.metric("30-Yr Mortgage Rate",f"{s['current_rate']}%")
    c5.metric("12-Mo Avg Sales",    f"{s['avg_sales_12mo']:,.0f}")

    viz = sys["visualizer"]
    st.plotly_chart(viz.active_vs_sales_trend(),    use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(viz.disparity_chart(),          use_container_width=True)
    with col_right:
        st.plotly_chart(viz.mortgage_rate_history(),    use_container_width=True)

    st.plotly_chart(viz.yoy_sales_comparison(),         use_container_width=True)
    st.plotly_chart(viz.economic_indicators_panel(),    use_container_width=True)
    st.plotly_chart(viz.correlation_heatmap(),          use_container_width=True)


# ── Tab 2: AI Agent Chat ──────────────────────────────────────────────────────

def render_chat(sys: dict) -> None:
    st.header("AI Real Estate Analyst")
    st.caption(
        "Ask about the Dayton MSA market, forecasts, mortgage rates, refinancing "
        "opportunities, seasonal patterns, or anything in the data."
    )

    chain = sys["chain"]

    # Initialise chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask a market question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analysing…"):
                answer = chain.ask(prompt, session_id="streamlit")
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Clear button
    if st.session_state.messages:
        if st.button("Clear conversation", key="clear_chat"):
            chain.clear_history("streamlit")
            st.session_state.messages = []
            st.rerun()


# ── Tab 3: Forecasts & Analysis ───────────────────────────────────────────────

@st.cache_resource(show_spinner="Running forecast models (this takes a minute)…")
def run_forecasts(_forecaster: MarketForecaster, _refi: RefiAnalyzer):
    """Cache-busted by forecaster object identity.  Runs once per session."""
    fc_sales,  ci_sales  = _forecaster.forecast_sarimax_sales()
    fc_active, ci_active = _forecaster.forecast_sarimax_active()
    mse_dict = _forecaster.evaluate_models("Sales")
    refi_df  = _refi.find_refi_windows()
    return fc_sales, ci_sales, fc_active, ci_active, mse_dict, refi_df


def render_forecasts(sys: dict) -> None:
    st.header("Forecasts & Analysis")

    forecaster = sys["forecaster"]
    refi       = sys["refi"]
    viz        = sys["visualizer"]

    with st.spinner("Fitting forecast models…"):
        fc_sales, ci_sales, fc_active, ci_active, mse_dict, refi_df = run_forecasts(
            forecaster, refi
        )

    # Forecast charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(viz.sales_forecast_chart(fc_sales, ci_sales), use_container_width=True)
    with col2:
        st.plotly_chart(viz.active_forecast_chart(fc_active, ci_active), use_container_width=True)

    # Model comparison
    if mse_dict:
        st.plotly_chart(viz.model_comparison_chart(mse_dict), use_container_width=True)
        st.caption(
            "MSE is computed on a held-out 12-month test set (last 12 observations), "
            "not on training data — giving a fair comparison of out-of-sample accuracy."
        )

    # Refi analysis
    st.subheader("Refinancing Opportunity Analysis")
    threshold = st.slider(
        "Rate threshold above current rate (pp)",
        min_value=0.25, max_value=2.0, value=config.REFI_THRESHOLD_PP, step=0.25,
    )
    refi_df_custom = refi.find_refi_windows(threshold_pp=threshold)
    st.plotly_chart(
        viz.refi_opportunity_windows(refi_df_custom, threshold_pp=threshold),
        use_container_width=True,
    )

    if not refi_df_custom.empty:
        st.dataframe(
            refi_df_custom.rename(columns={
                "date": "Month", "rate": "Rate (%)", "excess_pp": "Excess (pp)",
                **{f"savings_{int(l/1000)}k": f"${l:,} Loan Savings/Mo"
                   for l in config.REFI_LOAN_AMOUNTS},
            }).style.format({
                "Rate (%)": "{:.2f}",
                "Excess (pp)": "{:.2f}",
                **{f"${l:,} Loan Savings/Mo": "${:,.0f}" for l in config.REFI_LOAN_AMOUNTS},
            }),
            use_container_width=True,
        )

    # Narrative summary from agent
    st.subheader("Refi Summary")
    st.info(refi.generate_refi_summary(threshold_pp=threshold))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key, model, init_clicked = render_sidebar()

    if init_clicked:
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
            st.stop()
        # Clear cached system so it rebuilds with new key/model
        st.cache_resource.clear()
        st.session_state.system = build_system(api_key, model)
        st.rerun()

    # Auto-init if key is available and system not yet loaded
    if "system" not in st.session_state and api_key:
        try:
            st.session_state.system = build_system(api_key, model)
        except Exception as e:
            st.error(f"Initialisation failed: {e}")
            st.stop()

    if "system" not in st.session_state:
        st.info("Enter your OpenAI API key in the sidebar and click **Initialize System**.")
        st.stop()

    system = st.session_state.system

    tab1, tab2, tab3 = st.tabs(["📊 Market Dashboard", "🤖 AI Analyst Chat", "📈 Forecasts & Analysis"])
    with tab1:
        render_dashboard(system)
    with tab2:
        render_chat(system)
    with tab3:
        render_forecasts(system)


if __name__ == "__main__":
    main()
