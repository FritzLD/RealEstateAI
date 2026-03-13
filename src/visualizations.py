"""
RealEstateVisualizer
All charts return plotly.graph_objects.Figure for Streamlit rendering.

Replaces the original matplotlib/seaborn charts with interactive Plotly
equivalents that support hover, zoom, and pan out of the box.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import config

_PALETTE  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
_TEMPLATE = "plotly_white"


class RealEstateVisualizer:
    """
    Parameters
    ----------
    df : Merged DataFrame from RealEstateDataLoader (index = monthly DatetimeIndex)
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    # ── Layout helper ─────────────────────────────────────────────────────────

    def _layout(
        self,
        fig: go.Figure,
        title: str,
        height: int = 420,
        **kwargs,
    ) -> go.Figure:
        fig.update_layout(
            title=dict(text=title, font=dict(size=15)),
            template=_TEMPLATE,
            height=height,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=20, t=60, b=50),
            **kwargs,
        )
        return fig

    # ── 1. Active listings vs Sales trend ────────────────────────────────────

    def active_vs_sales_trend(self) -> go.Figure:
        """Dual-axis: Sales bars + Active Listings line with moving averages."""
        df = self.df
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(x=df.index, y=df["Sales"], name="Monthly Sales",
                   marker_color=_PALETTE[0], opacity=0.75),
            secondary_y=False,
        )

        if f"MA_{config.MA_LONG}" in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[f"MA_{config.MA_LONG}"],
                           name=f"{config.MA_LONG}-Month Sales MA",
                           line=dict(color=_PALETTE[2], width=2, dash="dash")),
                secondary_y=False,
            )

        fig.add_trace(
            go.Scatter(x=df.index, y=df["Active"], name="Active Listings",
                       line=dict(color=_PALETTE[1], width=2)),
            secondary_y=True,
        )

        fig.update_yaxes(title_text="Sales (units)", secondary_y=False)
        fig.update_yaxes(title_text="Active Listings", secondary_y=True)
        return self._layout(fig, "Dayton MSA – Monthly Sales vs Active Listings", height=450)

    # ── 2. Sales forecast chart ───────────────────────────────────────────────

    def sales_forecast_chart(
        self,
        forecast: pd.Series,
        conf_int: pd.DataFrame,
    ) -> go.Figure:
        """Historical sales + SARIMAX 12-month forecast with CI band."""
        df = self.df
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df.index, y=df["Sales"],
            name="Historical Sales", line=dict(color=_PALETTE[0], width=2),
        ))

        # Confidence interval band
        fig.add_trace(go.Scatter(
            x=list(conf_int.index) + list(conf_int.index[::-1]),
            y=list(conf_int["upper"]) + list(conf_int["lower"][::-1]),
            fill="toself", fillcolor="rgba(255,127,14,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% Confidence Interval", showlegend=True,
        ))

        fig.add_trace(go.Scatter(
            x=forecast.index, y=forecast.values,
            name="SARIMAX Forecast", line=dict(color=_PALETTE[1], width=2, dash="dot"),
            mode="lines+markers",
        ))

        return self._layout(
            fig,
            f"Sales Forecast – {forecast.index[-1].strftime('%B %Y')} (SARIMAX)",
            height=430,
        )

    # ── 3. Active listings forecast chart ────────────────────────────────────

    def active_forecast_chart(
        self,
        forecast: pd.Series,
        conf_int: pd.DataFrame,
    ) -> go.Figure:
        """Historical active listings + SARIMAX 12-month forecast with CI band."""
        df = self.df
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df.index, y=df["Active"],
            name="Historical Active Listings", line=dict(color=_PALETTE[1], width=2),
        ))

        fig.add_trace(go.Scatter(
            x=list(conf_int.index) + list(conf_int.index[::-1]),
            y=list(conf_int["upper"]) + list(conf_int["lower"][::-1]),
            fill="toself", fillcolor="rgba(44,160,44,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% Confidence Interval", showlegend=True,
        ))

        fig.add_trace(go.Scatter(
            x=forecast.index, y=forecast.values,
            name="SARIMAX Forecast", line=dict(color=_PALETTE[2], width=2, dash="dot"),
            mode="lines+markers",
        ))

        return self._layout(
            fig,
            f"Active Listings Forecast – {forecast.index[-1].strftime('%B %Y')} (SARIMAX)",
            height=430,
        )

    # ── 3b. All-models forecast chart ────────────────────────────────────────

    def all_models_forecast_chart(self, saved: dict) -> go.Figure:
        """
        All sales-model forecasts overlaid on historical Sales bars.
        Highlights the best model (★); CI bands shown for SARIMAX and Prophet only.

        Parameters
        ----------
        saved : dict loaded from data/forecasts/forecasts.json
        """
        df  = self.df
        fig = go.Figure()

        # Historical sales as muted gray bars
        fig.add_trace(go.Bar(
            x=df.index, y=df["Sales"],
            name="Historical Sales",
            marker_color="rgba(150,150,150,0.45)",
        ))

        # Per-model color pairs: (line color, CI fill rgba)
        _MCOLORS = {
            "SARIMAX":  ("#1f77b4", "rgba(31,119,180,0.12)"),
            "Prophet":  ("#ff7f0e", "rgba(255,127,14,0.12)"),
            "DNN":      ("#2ca02c", "rgba(44,160,44,0.12)"),
            "CNN":      ("#d62728", "rgba(214,39,40,0.12)"),
            "LSTM":     ("#9467bd", "rgba(148,103,189,0.12)"),
            "CNN_LSTM": ("#8c564b", "rgba(140,86,75,0.12)"),
        }

        best_model  = saved.get("best_model", "")
        models_dict = saved.get("models", {})

        # Sales-target models only — skip SARIMAX_Active (active listings)
        sales_models = [k for k in models_dict if k != "SARIMAX_Active"]

        for name in sales_models:
            m     = models_dict[name]
            dates = pd.to_datetime(m["dates"])
            fc    = m["forecast"]
            color, fill = _MCOLORS.get(name, ("#555555", "rgba(85,85,85,0.12)"))
            is_best = (name == best_model)

            # CI shaded band — only for models that include lower/upper bounds
            if "lower" in m and "upper" in m:
                upper = m["upper"]
                lower = m["lower"]
                fig.add_trace(go.Scatter(
                    x=list(dates) + list(dates[::-1]),
                    y=upper + lower[::-1],
                    fill="toself", fillcolor=fill,
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"{name} 80% CI", showlegend=False,
                    hoverinfo="skip",
                ))

            label = f"★ {name} (best)" if is_best else name
            fig.add_trace(go.Scatter(
                x=dates, y=fc,
                name=label,
                line=dict(
                    color=color,
                    width=3.5 if is_best else 1.8,
                    dash="solid" if is_best else "dot",
                ),
                mode="lines+markers" if is_best else "lines",
            ))

        # Vertical divider separating history from forecast horizon
        # Note: pass x as an ISO string — Plotly's add_vline annotation code
        # cannot do arithmetic on a raw pandas Timestamp (raises TypeError).
        if sales_models:
            first_date = pd.to_datetime(models_dict[sales_models[0]]["dates"][0])
            fig.add_vline(
                x=first_date.strftime("%Y-%m-%d"),
                line_dash="dash", line_color="gray", line_width=1,
                annotation_text="Forecast →",
                annotation_position="top right",
            )

        data_through = saved.get("data_through", "")
        title = "12-Month Sales Forecast – All Models"
        if data_through:
            title += f"  (data through {data_through})"

        fig.update_yaxes(title_text="Sales (units)")
        return self._layout(fig, title, height=520)

    # ── 4. Market disparity chart ─────────────────────────────────────────────

    def disparity_chart(self) -> go.Figure:
        """Disparity % with short and long MAs and balanced-market threshold line."""
        df = self.df
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df.index, y=df["Disparity_pct"],
            name="Disparity %", line=dict(color=_PALETTE[3], width=1.5),
            opacity=0.6,
        ))

        if "Disparity_MA" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["Disparity_MA"],
                name=f"{config.MA_SHORT}-Month Disparity MA",
                line=dict(color=_PALETTE[0], width=2),
            ))

        # Balanced-market baseline = historical mean disparity %
        # Uses actual Dayton MSA data average to remove long-run bias
        mean_disparity = float(df["Disparity_pct"].mean())
        fig.add_hline(
            y=mean_disparity, line_dash="dash", line_color="green",
            annotation_text=f"Historical Mean ({mean_disparity:.1f}%)",
            annotation_position="bottom right",
        )

        fig.update_yaxes(title_text="Disparity % (Active−Sales)/Active")
        return self._layout(fig, "Dayton MSA – Market Inventory Disparity", height=400)

    # ── 5. Mortgage rate history ──────────────────────────────────────────────

    def mortgage_rate_history(self) -> go.Figure:
        """30-year FRM history with current rate annotation."""
        if "30yrFRM" not in self.df.columns:
            return go.Figure()
        rates = self.df["30yrFRM"].dropna()
        current = float(rates.iloc[-1])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rates.index, y=rates.values,
            name="30-Year Fixed Rate", line=dict(color=_PALETTE[0], width=2),
            fill="tozeroy", fillcolor="rgba(31,119,180,0.10)",
        ))

        fig.add_hline(
            y=current, line_dash="dot", line_color=_PALETTE[3],
            annotation_text=f"Current: {current:.2f}%",
            annotation_position="top right",
        )

        fig.update_yaxes(title_text="Rate (%)", ticksuffix="%")
        return self._layout(fig, "30-Year Fixed Mortgage Rate History", height=380)

    # ── 6. Refinancing opportunity windows ───────────────────────────────────

    def refi_opportunity_windows(
        self,
        refi_df: pd.DataFrame,
        threshold_pp: float = config.REFI_THRESHOLD_PP,
    ) -> go.Figure:
        """Rate chart with shaded windows where refinancing makes sense."""
        if "30yrFRM" not in self.df.columns:
            return go.Figure()

        rates   = self.df["30yrFRM"].dropna()
        current = float(rates.iloc[-1])
        trigger = current + threshold_pp

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rates.index, y=rates.values,
            name="30-Year Fixed Rate", line=dict(color=_PALETTE[0], width=2),
        ))

        # Shade refi-opportunity months
        if not refi_df.empty:
            for _, row in refi_df.iterrows():
                fig.add_vrect(
                    x0=row["date"] - pd.DateOffset(days=15),
                    x1=row["date"] + pd.DateOffset(days=15),
                    fillcolor="rgba(214,39,40,0.15)", line_width=0,
                )

        fig.add_hline(
            y=trigger, line_dash="dash", line_color=_PALETTE[3],
            annotation_text=f"Refi trigger ({trigger:.2f}%)",
            annotation_position="top left",
        )
        fig.add_hline(
            y=current, line_dash="dot", line_color=_PALETTE[2],
            annotation_text=f"Current rate ({current:.2f}%)",
            annotation_position="bottom right",
        )

        fig.update_yaxes(title_text="Rate (%)", ticksuffix="%")
        note = f"Red bands = months where refinancing from that rate saves ≥ {threshold_pp} pp"
        return self._layout(
            fig, f"Refinancing Opportunity Windows – {note}", height=400
        )

    # ── 7. Correlation heatmap ────────────────────────────────────────────────

    def correlation_heatmap(self) -> go.Figure:
        """Annotated correlation matrix for all numeric columns."""
        num_cols = [
            c for c in ["Active", "Sales", "30yrFRM", "CPI", "Inflation_Rate",
                         "Unemployment", "Consumer_Sentiment_Index", "GPRHC_USA",
                         "Disparity_pct"]
            if c in self.df.columns
        ]
        corr = self.df[num_cols].corr().round(2)

        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr.values,
            texttemplate="%{text:.2f}",
            hoverongaps=False,
        ))

        return self._layout(fig, "Variable Correlation Matrix", height=500)

    # ── 8. Economic indicators panel ─────────────────────────────────────────

    def economic_indicators_panel(self) -> go.Figure:
        """2×2 subplot for CPI, Inflation Rate, Unemployment, Consumer Sentiment."""
        indicators = [
            ("CPI",                    "CPI",                "#1f77b4"),
            ("Inflation_Rate",         "Inflation Rate (%)", "#ff7f0e"),
            ("Unemployment",           "Unemployment (%)",   "#2ca02c"),
            ("Consumer_Sentiment_Index","Consumer Sentiment","#9467bd"),
        ]
        available = [(col, lbl, clr) for col, lbl, clr in indicators if col in self.df.columns]
        if not available:
            return go.Figure()

        rows = (len(available) + 1) // 2
        fig  = make_subplots(rows=rows, cols=2, subplot_titles=[lbl for _, lbl, _ in available])

        for i, (col, lbl, clr) in enumerate(available):
            r, c = divmod(i, 2)
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df[col].dropna(),
                    name=lbl,
                    line=dict(color=clr, width=1.5),
                    showlegend=False,
                ),
                row=r + 1, col=c + 1,
            )

        fig.update_layout(
            title=dict(text="Economic Indicators", font=dict(size=15)),
            template=_TEMPLATE,
            height=480,
            margin=dict(l=50, r=20, t=60, b=50),
        )
        return fig

    # ── 9. Model comparison chart ─────────────────────────────────────────────

    def model_comparison_chart(self, mse_dict: Dict[str, float]) -> go.Figure:
        """Horizontal bar chart of test-set MSE by model (lower = better)."""
        if not mse_dict:
            return go.Figure()

        sorted_items = sorted(mse_dict.items(), key=lambda x: x[1])
        names  = [k for k, _ in sorted_items]
        values = [v for _, v in sorted_items]
        colors = [_PALETTE[0] if i == 0 else _PALETTE[4] for i in range(len(names))]

        fig = go.Figure(go.Bar(
            x=values, y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:,.0f}" for v in values],
            textposition="outside",
        ))

        fig.update_xaxes(title_text="Test-Set MSE (lower is better)")
        return self._layout(
            fig, "Forecast Model Comparison (Held-Out 12-Month Test MSE)", height=300
        )

    # ── 10. Year-over-year sales comparison ──────────────────────────────────

    def yoy_sales_comparison(self) -> go.Figure:
        """Grouped bar: current year monthly sales vs prior year."""
        df = self.df
        latest_year = df.index.year.max()
        current = df[df.index.year == latest_year]["Sales"]
        prior   = df[df.index.year == latest_year - 1]["Sales"]

        month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[month_labels[m - 1] for m in prior.index.month],
            y=prior.values, name=str(latest_year - 1),
            marker_color=_PALETTE[4], opacity=0.8,
        ))
        fig.add_trace(go.Bar(
            x=[month_labels[m - 1] for m in current.index.month],
            y=current.values, name=str(latest_year),
            marker_color=_PALETTE[0],
        ))

        fig.update_layout(barmode="group")
        fig.update_yaxes(title_text="Sales (units)")
        return self._layout(
            fig, f"Year-over-Year Sales Comparison ({latest_year - 1} vs {latest_year})",
            height=380,
        )
