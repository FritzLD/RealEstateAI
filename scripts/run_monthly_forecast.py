"""
run_monthly_forecast.py
=======================
Run this script locally in VS Code once a month after updating your data files.
It fits all 6 models and saves forecast results to data/forecasts/forecasts.json,
which the Streamlit Cloud app loads instantly (no model fitting on the cloud).

Usage:
    cd "D:\\Real estate Forecast\\RealEstateAI"
    pip install -r requirements-local.txt   # first time only (installs tensorflow)
    python scripts/run_monthly_forecast.py

Output:
    data/forecasts/forecasts.json   ← commit this file and push to GitHub
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

from src import config
from src.data_loader import RealEstateDataLoader
from src.forecasting import MarketForecaster, save_forecasts

# ── Constants ─────────────────────────────────────────────────────────────────

HORIZON     = config.FORECAST_HORIZON   # 12 months
SEQ_LEN     = config.SEQ_LENGTH         # 12-month lookback for neural models
EPOCHS      = 50
BATCH_SIZE  = 16
TEST_N      = config.TEST_HOLDOUT       # last 12 months held out for MSE

# Features used by neural models (same as original notebooks)
NEURAL_FEATURES = [
    "Active", "CPI", "New_Listings", "Sales_Volume", "Expired_Listings",
    "30yrFRM", "Inflation_Rate", "Unemployment", "Consumer_Sentiment_Index",
]
TARGET = "Sales"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _available_features(df: pd.DataFrame) -> list[str]:
    """Return only the NEURAL_FEATURES columns that actually exist in df."""
    return [f for f in NEURAL_FEATURES if f in df.columns]


def _make_sequences(df: pd.DataFrame, features: list[str], target: str, seq_len: int):
    """
    Build (X, y) arrays for supervised sequence learning.
    X shape: (n_samples, seq_len, n_features)
    y shape: (n_samples,)
    """
    from sklearn.preprocessing import MinMaxScaler

    feat_data   = df[features].values
    target_data = df[target].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    feat_scaled   = scaler_X.fit_transform(feat_data)
    target_scaled = scaler_y.fit_transform(target_data.reshape(-1, 1)).flatten()

    X, y = [], []
    for i in range(seq_len, len(feat_scaled)):
        X.append(feat_scaled[i - seq_len : i])
        y.append(target_scaled[i])

    return np.array(X), np.array(y), scaler_X, scaler_y


def _rolling_forecast(model, last_window: np.ndarray, scaler_X, scaler_y,
                      steps: int, n_features: int) -> np.ndarray:
    """
    Generate multi-step ahead forecast by rolling the window forward.
    For each step the predicted Sales value is inserted back at position 0
    of the feature vector (matching the original notebook approach).
    """
    forecast    = []
    curr_window = last_window.copy()   # shape: (seq_len, n_features)

    for _ in range(steps):
        pred_scaled = model.predict(curr_window.reshape(1, *curr_window.shape), verbose=0)[0][0]
        forecast.append(pred_scaled)

        # Build next row: set feature-0 (Sales) to prediction, keep others constant
        next_row         = np.zeros(n_features)
        next_row[0]      = pred_scaled
        next_row[1:]     = curr_window[-1, 1:]   # carry last known values forward
        curr_window      = np.vstack([curr_window[1:], next_row])

    return scaler_y.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()


# ── Neural model builders ─────────────────────────────────────────────────────

def _build_dnn(seq_len: int, n_features: int):
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1),
    ])


def _build_cnn(seq_len: int, n_features: int):
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),
        tf.keras.layers.Conv1D(64, kernel_size=2, activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(32, kernel_size=2, activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1),
    ])


def _build_lstm(seq_len: int, n_features: int):
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1),
    ])


def _build_cnn_lstm(seq_len: int, n_features: int):
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),
        tf.keras.layers.Conv1D(64, kernel_size=2, activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1),
    ])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("RealEstateAI – Monthly Forecast Runner")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading market data...")
    loader    = RealEstateDataLoader()
    df        = loader.df
    data_thru = df.index[-1].strftime("%B %Y")
    fc_start  = (df.index[-1] + pd.DateOffset(months=1)).strftime("%B %Y")
    fc_dates  = [
        (df.index[-1] + pd.DateOffset(months=i + 1)).strftime("%Y-%m-%d")
        for i in range(HORIZON)
    ]
    print(f"    Data through: {data_thru}  |  Forecasting: {fc_start} (+{HORIZON} months)")

    results: dict = {
        "generated_at":  datetime.now().isoformat(),
        "data_through":  data_thru,
        "forecast_start": fc_start,
        "models":        {},
        "mse_scores":    {},
        "best_model":    None,
    }

    # ── 2. SARIMAX ────────────────────────────────────────────────────────────
    print("\n[2/4] Fitting SARIMAX + Prophet...")
    forecaster = MarketForecaster(df, loader.exog_cols)
    try:
        fc_s, ci_s = forecaster.forecast_sarimax_sales(steps=HORIZON)
        results["models"]["SARIMAX"] = {
            "dates":    fc_dates,
            "forecast": fc_s.values.tolist(),
            "lower":    ci_s["lower"].values.tolist(),
            "upper":    ci_s["upper"].values.tolist(),
        }
        print("    SARIMAX Sales ✓")
    except Exception as e:
        print(f"    SARIMAX Sales ✗  ({e})")

    try:
        fc_a, ci_a = forecaster.forecast_sarimax_active(steps=HORIZON)
        results["models"]["SARIMAX_Active"] = {
            "dates":    fc_dates,
            "forecast": fc_a.values.tolist(),
            "lower":    ci_a["lower"].values.tolist(),
            "upper":    ci_a["upper"].values.tolist(),
        }
        print("    SARIMAX Active ✓")
    except Exception as e:
        print(f"    SARIMAX Active ✗  ({e})")

    # ── 3. Prophet ────────────────────────────────────────────────────────────
    try:
        fc_p = forecaster.forecast_prophet_sales(steps=HORIZON)
        results["models"]["Prophet"] = {
            "dates":    fc_dates,
            "forecast": fc_p["yhat"].values.tolist(),
            "lower":    fc_p["yhat_lower"].values.tolist(),
            "upper":    fc_p["yhat_upper"].values.tolist(),
        }
        print("    Prophet ✓")
    except Exception as e:
        print(f"    Prophet ✗  ({e})")

    # ── 4. Neural models ──────────────────────────────────────────────────────
    print("\n[3/4] Fitting neural models (DNN / CNN / LSTM / CNN-LSTM)...")
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        features  = _available_features(df)
        n_feat    = len(features)
        print(f"    Features ({n_feat}): {features}")

        # Drop rows where any feature or target is NaN
        model_df = df[features + [TARGET]].dropna()

        X, y, scaler_X, scaler_y = _make_sequences(model_df, features, TARGET, SEQ_LEN)

        # Train / test split
        split    = len(X) - TEST_N
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        last_window = X[-1]   # most recent window for rolling forecast

        neural_builders = {
            "DNN":      _build_dnn,
            "CNN":      _build_cnn,
            "LSTM":     _build_lstm,
            "CNN_LSTM": _build_cnn_lstm,
        }

        for name, builder in neural_builders.items():
            try:
                model = builder(SEQ_LEN, n_feat)
                model.compile(optimizer="adam", loss="mean_squared_error")
                model.fit(
                    X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test),
                    verbose=0,
                )
                # MSE on held-out test set
                preds_scaled = model.predict(X_test, verbose=0).flatten()
                preds        = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
                actual       = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
                mse          = float(np.mean((preds - actual) ** 2))

                # 12-month rolling forecast
                fc_vals = _rolling_forecast(model, last_window, scaler_X, scaler_y,
                                            HORIZON, n_feat)

                results["models"][name]   = {"dates": fc_dates, "forecast": fc_vals.tolist()}
                results["mse_scores"][name] = round(mse, 1)
                print(f"    {name} ✓  (test MSE: {mse:,.0f})")

            except Exception as e:
                print(f"    {name} ✗  ({e})")

        # Add SARIMAX + Prophet MSE scores from evaluate_models
        eval_mse = forecaster.evaluate_models("Sales")
        for k, v in eval_mse.items():
            results["mse_scores"][k] = round(v, 1)

    except ImportError:
        print("    TensorFlow not installed — skipping neural models.")
        print("    Run: pip install -r requirements-local.txt")

    # ── 5. Pick best model by MSE ─────────────────────────────────────────────
    if results["mse_scores"]:
        results["best_model"] = min(results["mse_scores"], key=results["mse_scores"].get)
        print(f"\n    Best model: {results['best_model']} "
              f"(MSE {results['mse_scores'][results['best_model']]:,.0f})")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    print("\n[4/4] Saving forecasts...")
    save_forecasts(results)
    out = config.DATA_DIR / "forecasts" / "forecasts.json"
    print(f"    Saved → {out}")

    print("\n" + "=" * 60)
    print("Done! Next steps:")
    print("  git add data/forecasts/forecasts.json")
    print(f'  git commit -m "Monthly forecast update – {data_thru}"')
    print("  git push")
    print("=" * 60)


if __name__ == "__main__":
    main()
