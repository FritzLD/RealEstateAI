"""
MarketForecaster
Encapsulates SARIMAX and Prophet forecasting with the logical bugs from the
original notebooks corrected.

Key fixes applied
─────────────────
1. Forecast start date is always derived dynamically from the last observed date,
   never hardcoded.
2. Future exogenous values for SARIMAX use the historical column means rather than
   repeating the last N observations (which silently used past data as the "future").
3. Model evaluation uses a held-out test set (last TEST_HOLDOUT months) so MSE is
   computed on unseen data, not training data.
4. DatetimeIndex frequency is verified before fitting; re-inferred if missing.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src import config

# Suppress harmless convergence/frequency warnings during fitting
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class MarketForecaster:
    """
    Fits SARIMAX and Prophet models for Sales and Active Listings.

    Parameters
    ----------
    df        : Merged DataFrame from RealEstateDataLoader (index = monthly DatetimeIndex)
    exog_cols : List of exogenous column names that have passed VIF screening
    """

    def __init__(self, df: pd.DataFrame, exog_cols: list[str] | None = None) -> None:
        self.df        = df.copy()
        self.exog_cols = exog_cols or []
        self._ensure_freq()

        # Fitted model objects (populated on demand)
        self._sarimax_sales:  object | None = None
        self._sarimax_active: object | None = None
        self._prophet_sales:  object | None = None
        self._prophet_active: object | None = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_freq(self) -> None:
        """Set 'MS' frequency on the index if missing (fixes VAR/SARIMAX warning)."""
        if self.df.index.freq is None:
            self.df.index = pd.DatetimeIndex(self.df.index).to_period("M").to_timestamp("MS")
            self.df.index.freq = "MS"

    @staticmethod
    def _safe_log(series: pd.Series) -> pd.Series:
        """Log-transform with small-constant guard for zeros/negatives."""
        floor = series[series > 0].min() * 0.1 if (series > 0).any() else 1.0
        return np.log(series.clip(lower=floor))

    def _get_exog(self, df: pd.DataFrame | None = None) -> pd.DataFrame | None:
        """Return the exogenous DataFrame aligned to the given (or full) df."""
        if not self.exog_cols:
            return None
        src = df if df is not None else self.df
        exog = src[self.exog_cols].dropna()
        return exog if not exog.empty else None

    def _future_exog(self, steps: int) -> pd.DataFrame | None:
        """
        Build a DataFrame of future exogenous values.

        Fix: use column means over all available history (flat-line assumption),
        NOT the last N rows (which would leak historical data into the forecast).
        """
        if not self.exog_cols:
            return None
        means = self.df[self.exog_cols].mean()
        future_index = pd.date_range(
            start=self.df.index[-1] + pd.DateOffset(months=1),
            periods=steps,
            freq="MS",
        )
        return pd.DataFrame(
            {col: means[col] for col in self.exog_cols},
            index=future_index,
        )

    def _forecast_index(self, steps: int) -> pd.DatetimeIndex:
        """Dynamic forecast start: always one month after the last observed date."""
        return pd.date_range(
            start=self.df.index[-1] + pd.DateOffset(months=1),
            periods=steps,
            freq="MS",
        )

    # ── SARIMAX ───────────────────────────────────────────────────────────────

    def _fit_sarimax(self, series: pd.Series) -> object:
        """Auto-select SARIMAX order via pmdarima.auto_arima and fit."""
        try:
            import pmdarima as pm
        except ImportError as e:
            raise ImportError("pmdarima is required for SARIMAX forecasting.") from e

        log_series = self._safe_log(series.dropna())
        exog = self._get_exog()
        if exog is not None:
            exog = exog.reindex(log_series.index).dropna()
            log_series = log_series.reindex(exog.index)

        auto = pm.auto_arima(
            log_series,
            exogenous=exog,
            m=12,
            seasonal=True,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            information_criterion="aic",
            maxiter=50,
        )
        order         = auto.order
        seasonal_order = auto.seasonal_order

        from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX

        model = _SARIMAX(
            log_series,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False)

    def fit_sarimax_sales(self) -> None:
        self._sarimax_sales = self._fit_sarimax(self.df["Sales"])

    def fit_sarimax_active(self) -> None:
        self._sarimax_active = self._fit_sarimax(self.df["Active"])

    def forecast_sarimax_sales(
        self, steps: int = config.FORECAST_HORIZON
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Returns (forecast_series, conf_int_df) in original (non-log) scale.
        conf_int_df has columns ['lower', 'upper'].
        """
        if self._sarimax_sales is None:
            self.fit_sarimax_sales()
        return self._sarimax_forecast(self._sarimax_sales, steps)

    def forecast_sarimax_active(
        self, steps: int = config.FORECAST_HORIZON
    ) -> Tuple[pd.Series, pd.DataFrame]:
        if self._sarimax_active is None:
            self.fit_sarimax_active()
        return self._sarimax_forecast(self._sarimax_active, steps)

    def _sarimax_forecast(
        self, fitted_model: object, steps: int
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Run in-sample forecast and back-transform from log scale."""
        future_exog = self._future_exog(steps)
        raw = fitted_model.get_forecast(steps=steps, exog=future_exog)

        idx = self._forecast_index(steps)
        fc  = pd.Series(np.exp(raw.predicted_mean.values), index=idx, name="forecast")

        ci_raw = raw.conf_int()
        ci = pd.DataFrame(
            {"lower": np.exp(ci_raw.iloc[:, 0].values),
             "upper": np.exp(ci_raw.iloc[:, 1].values)},
            index=idx,
        )
        return fc, ci

    # ── Prophet ───────────────────────────────────────────────────────────────

    def _fit_prophet(self, series: pd.Series) -> object:
        try:
            from prophet import Prophet
        except ImportError as e:
            raise ImportError("prophet is required for Prophet forecasting.") from e

        train = series.dropna().reset_index()
        train.columns = ["ds", "y"]
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.80,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(train)
        return m

    def fit_prophet_sales(self) -> None:
        self._prophet_sales = self._fit_prophet(self.df["Sales"])

    def fit_prophet_active(self) -> None:
        self._prophet_active = self._fit_prophet(self.df["Active"])

    def forecast_prophet_sales(
        self, steps: int = config.FORECAST_HORIZON
    ) -> pd.DataFrame:
        if self._prophet_sales is None:
            self.fit_prophet_sales()
        return self._prophet_forecast(self._prophet_sales, steps)

    def forecast_prophet_active(
        self, steps: int = config.FORECAST_HORIZON
    ) -> pd.DataFrame:
        if self._prophet_active is None:
            self.fit_prophet_active()
        return self._prophet_forecast(self._prophet_active, steps)

    def _prophet_forecast(self, fitted_model: object, steps: int) -> pd.DataFrame:
        """Returns DataFrame with columns ds, yhat, yhat_lower, yhat_upper."""
        future = fitted_model.make_future_dataframe(periods=steps, freq="MS")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = fitted_model.predict(future)
        cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
        result = forecast[cols].tail(steps).copy()
        result = result.set_index("ds")
        result.index = pd.DatetimeIndex(result.index, freq="MS")
        return result

    # ── Model evaluation ──────────────────────────────────────────────────────

    def evaluate_models(
        self, target: str = "Sales", test_n: int = config.TEST_HOLDOUT
    ) -> Dict[str, float]:
        """
        Fit each model on df[:-test_n], forecast test_n steps, compute MSE
        against the held-out df[-test_n:] values.  Returns {model_name: mse}.

        Fix: evaluation uses held-out TEST data, not training data.
        """
        if len(self.df) <= test_n + 12:
            return {}

        train_df = self.df.iloc[:-test_n].copy()
        actual   = self.df[target].iloc[-test_n:].values
        results: Dict[str, float] = {}

        # SARIMAX
        try:
            train_forecaster = MarketForecaster(train_df, self.exog_cols)
            if target == "Sales":
                fc, _ = train_forecaster.forecast_sarimax_sales(steps=test_n)
            else:
                fc, _ = train_forecaster.forecast_sarimax_active(steps=test_n)
            results["SARIMAX"] = float(np.mean((fc.values - actual) ** 2))
        except Exception:
            pass

        # Prophet
        try:
            train_forecaster = MarketForecaster(train_df, self.exog_cols)
            if target == "Sales":
                fc_df = train_forecaster.forecast_prophet_sales(steps=test_n)
            else:
                fc_df = train_forecaster.forecast_prophet_active(steps=test_n)
            results["Prophet"] = float(np.mean((fc_df["yhat"].values - actual) ** 2))
        except Exception:
            pass

        return results

    # ── Convenience ───────────────────────────────────────────────────────────

    def get_forecast_summary(self, steps: int = config.FORECAST_HORIZON) -> str:
        """Return a brief text summary of the SARIMAX 12-month outlook."""
        try:
            fc_sales,  _ = self.forecast_sarimax_sales(steps=steps)
            fc_active, _ = self.forecast_sarimax_active(steps=steps)
            horizon_end  = fc_sales.index[-1].strftime("%B %Y")
            return (
                f"12-month SARIMAX forecast through {horizon_end}:\n"
                f"- Projected average monthly sales: {fc_sales.mean():.0f} units\n"
                f"- Projected average active listings: {fc_active.mean():.0f} units\n"
                f"- End-of-horizon sales: {fc_sales.iloc[-1]:.0f} units\n"
                f"- End-of-horizon active listings: {fc_active.iloc[-1]:.0f} units\n"
                f"Note: exogenous variables held at historical mean values."
            )
        except Exception as exc:
            return f"Forecast unavailable: {exc}"
